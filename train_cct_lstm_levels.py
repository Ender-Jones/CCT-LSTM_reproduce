import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from model import CCT_LSTM_Model
from dataset import get_default_transforms
from file_path_gen import FilePathGen


"""
Stage 2: Multilevel classification (T1, T3-ctrl, T3-test), 5-fold stratified CV
-------------------------------------------------------------------------------
This script trains CCT-LSTM for the 3-class task: T1 vs. T3-ctrl vs. T3-test.
It uses subject-level stratified 5-fold CV to preserve ctrl/test ratio, and expects
Stage 1 best weights for both modalities per fold:
  weights/cct_landmark_fold_{k}_best.pth
  weights/cct_rppg_fold_{k}_best.pth

Data selection
--------------
- T1 sequences from all subjects -> label 0 (T1)
- T3 sequences from ctrl subjects -> label 1 (T3-ctrl)
- T3 sequences from test subjects -> label 2 (T3-test)

We build paired sequences only from window-ID intersections (same as VideoSequenceDataset)
and require at least MIN_SEQ_LENGTH images per pair.
"""


MIN_SEQ_LENGTH = 10


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_window_id(path: Path) -> int:
    try:
        return int(path.stem.split('_')[-1])
    except (IndexError, ValueError):
        return -1


class LevelsSequenceDataset(Dataset):
    def __init__(self, data_root: Path, subjects: List[str], subject_to_group: Dict[str, str], transform):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.subject_to_group = subject_to_group
        self.transform = transform
        self.sequences: List[Tuple[List[Path], List[Path], int]] = []

        for subject_id in self.subjects:
            group = self.subject_to_group.get(subject_id)
            if group not in {"ctrl", "test"}:
                # skip unknown
                continue

            # T1 -> label 0
            self._collect_sequences_for(subject_id, level_str="T1", label=0)

            # T3 -> label by group: ctrl->1, test->2
            label_t3 = 1 if group == "ctrl" else 2
            self._collect_sequences_for(subject_id, level_str="T3", label=label_t3)

    def _collect_sequences_for(self, subject_id: str, level_str: str, label: int):
        landmark_dir = self.data_root / subject_id / 'mtf_images' / 'landmark'
        rppg_dir = self.data_root / subject_id / 'mtf_images' / 'rppg'
        if not landmark_dir.is_dir() or not rppg_dir.is_dir():
            return

        lms = sorted(landmark_dir.glob(f'{subject_id}_{level_str}_window_*.png'))
        rps = sorted(rppg_dir.glob(f'{subject_id}_{level_str}_window_*.png'))
        if not lms or not rps:
            return

        lm_map = {wid: p for p in lms if (wid := _parse_window_id(p)) >= 0}
        rp_map = {wid: p for p in rps if (wid := _parse_window_id(p)) >= 0}
        common_ids = sorted(set(lm_map.keys()) & set(rp_map.keys()))
        if len(common_ids) < MIN_SEQ_LENGTH:
            return

        seq_landmark = [lm_map[i] for i in common_ids]
        seq_rppg = [rp_map[i] for i in common_ids]
        self.sequences.append((seq_landmark, seq_rppg, label))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        landmark_paths, rppg_paths, label = self.sequences[idx]

        landmark_tensors = []
        rppg_tensors = []
        for lp, rp in zip(landmark_paths, rppg_paths):
            li = Image.open(lp).convert('RGB')
            ri = Image.open(rp).convert('RGB')
            if self.transform:
                li = self.transform(li)
                ri = self.transform(ri)
            landmark_tensors.append(li)
            rppg_tensors.append(ri)

        landmark_sequence = torch.stack(landmark_tensors)  # (N, C, H, W)
        rppg_sequence = torch.stack(rppg_tensors)
        final_sequence = torch.stack([landmark_sequence, rppg_sequence], dim=1)  # (N, 2, C, H, W)
        return final_sequence, label


def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.unsqueeze(0)
    return x


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for sequences, labels in tqdm(loader, desc="Training", leave=False):
        sequences = _ensure_batch_dim(sequences).to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
    return {"train_loss": avg_loss, "train_acc": acc, "train_f1": f1}


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for sequences, labels in tqdm(loader, desc="Validating", leave=False):
        sequences = _ensure_batch_dim(sequences).to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        logits = model(sequences)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
    cm = confusion_matrix(all_labels, all_preds) if all_labels else np.zeros((3, 3), dtype=int)
    report = classification_report(all_labels, all_preds, digits=4) if all_labels else ""
    return {"val_loss": avg_loss, "val_acc": acc, "val_f1": f1}, cm, report


def get_data_root(cli_data_root: str | None) -> Path:
    if cli_data_root:
        p = Path(cli_data_root)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    fpg = FilePathGen()
    return fpg.datapath


def read_subject_groups(data_root: Path) -> Dict[str, str]:
    # master_manifest.csv is under: .../UBFC-Phys/master_manifest.csv
    manifest_path = data_root.parent / 'master_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    # Expect columns: subject, group ('ctrl' or 'test')
    mapping = {str(row['subject']).strip(): str(row['group']).strip() for _, row in df.iterrows()}
    return mapping


def main(args: argparse.Namespace):
    # seed init (support random seed if <0)
    def _init_and_set_seed(seed_arg: int) -> int:
        if seed_arg is None or seed_arg < 0:
            seed = np.random.randint(0, 2**31 - 1)
        else:
            seed = int(seed_arg)
        print(f"Using seed: {seed}")
        set_seed(seed)
        return seed

    used_seed = _init_and_set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_root = get_data_root(args.data_root)
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('s')],
                      key=lambda x: int(x[1:]))
    if not subjects:
        raise RuntimeError(f"No subject folders under {data_root}")

    subject_to_group = read_subject_groups(data_root)
    # build stratify labels per subject (ctrl/test) to preserve ratio across folds
    y_strat = [0 if subject_to_group.get(s) == 'ctrl' else 1 for s in subjects]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = vars(args).copy()
        wandb_config["used_seed"] = used_seed
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=used_seed)

    cct_params = dict(
        img_size=(224, 224),
        embedding_dim=256,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=6,
        num_heads=4,
        mlp_ratio=2.0,
        positional_embedding='learnable',
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    transform = get_default_transforms()

    fold_best_f1s: List[float] = []
    best_epochs: List[int] = []
    best_overfit_gaps: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, y_strat), start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold_idx}/5 (levels)")
        print("=" * 60)
        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")

        train_dataset = LevelsSequenceDataset(data_root, train_subjects, subject_to_group, transform)
        val_dataset = LevelsSequenceDataset(data_root, val_subjects, subject_to_group, transform)
        print(f"Sequences -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("[WARN] Empty dataset split; skipping this fold.")
            continue

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=(args.num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=(args.num_workers > 0)
        )

        model = CCT_LSTM_Model(cct_params=cct_params,
                                lstm_hidden_size=args.lstm_hidden_size,
                                lstm_num_layers=args.lstm_num_layers,
                                num_classes=3).to(device)

        # Load Stage 1 best weights
        weights_dir = Path(args.stage1_weights_dir)
        landmark_w = weights_dir / f"cct_landmark_fold_{fold_idx}_best.pth"
        rppg_w = weights_dir / f"cct_rppg_fold_{fold_idx}_best.pth"
        if not landmark_w.exists() or not rppg_w.exists():
            raise FileNotFoundError(
                f"Missing Stage1 weights for fold {fold_idx}:\n  {landmark_w}\n  {rppg_w}")
        model.load_pretrained_weights(str(landmark_w), str(rppg_w))

        # Optional: freeze CCT backbones
        if args.freeze_cct:
            for p in model.cct_landmark.parameters():
                p.requires_grad = False
            for p in model.cct_rppg.parameters():
                p.requires_grad = False
            model.cct_landmark.eval()
            model.cct_rppg.eval()
            optim_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
        else:
            optim_params = model.parameters()

        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_path = output_dir / f"cctlstm_levels_fold_{fold_idx}_best.pth"
        epochs_no_improve = 0
        best_epoch_for_fold = None
        best_overfit_gap_for_fold = None

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, cm, report = validate(model, val_loader, criterion, device)
            print(f"Train: loss={train_metrics['train_loss']:.4f} acc={train_metrics['train_acc']:.4f} f1={train_metrics['train_f1']:.4f}")
            print(f"Val  : loss={val_metrics['val_loss']:.4f} acc={val_metrics['val_acc']:.4f} f1={val_metrics['val_f1']:.4f}")

            scheduler.step(val_metrics['val_f1'])
            if wandb_run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "train_loss": train_metrics["train_loss"],
                        "train_acc": train_metrics["train_acc"],
                        "train_f1": train_metrics["train_f1"],
                        "val_loss": val_metrics["val_loss"],
                        "val_acc": val_metrics["val_acc"],
                        "val_f1": val_metrics["val_f1"],
                        "overfit_gap": train_metrics["train_acc"] - val_metrics["val_acc"],
                        "learning_rate": current_lr,
                    }
                )
            if val_metrics['val_f1'] > best_val_f1 + 1e-6:
                best_val_f1 = val_metrics['val_f1']
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                fold_dir = Path(args.report_dir) / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                np.save(fold_dir / "confusion_matrix.npy", cm)
                with open(fold_dir / "classification_report.txt", "w") as f:
                    f.write(report)
                print(f"[BEST] Saved model to {best_path} (val_f1={best_val_f1:.4f})")
                best_epoch_for_fold = epoch
                best_overfit_gap_for_fold = train_metrics["train_acc"] - val_metrics["val_acc"]
                if wandb_run is not None:
                    wandb.run.summary[f"best_val_f1_fold_{fold_idx}"] = float(best_val_f1)
                    wandb.run.summary[f"best_epoch_fold_{fold_idx}"] = best_epoch_for_fold
                    wandb.run.summary[f"overfit_gap_fold_{fold_idx}"] = best_overfit_gap_for_fold
                    wandb.run.summary[f"confusion_matrix_fold_{fold_idx}"] = cm.tolist()
                    wandb.run.summary[f"classification_report_fold_{fold_idx}"] = report
            else:
                epochs_no_improve += 1

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
                break

        print(f"Fold {fold_idx} best val_f1: {best_val_f1:.4f}")
        if best_val_f1 >= 0.0:
            fold_best_f1s.append(best_val_f1)
        if best_epoch_for_fold is not None:
            best_epochs.append(best_epoch_for_fold)
        if best_overfit_gap_for_fold is not None:
            best_overfit_gaps.append(best_overfit_gap_for_fold)

    if fold_best_f1s:
        avg_best_f1 = float(np.mean(fold_best_f1s))
        print("\n" + "=" * 60)
        print(f"Average best val_f1 across folds: {avg_best_f1:.4f}")
        if wandb_run is not None:
            wandb.run.summary["avg_val_f1_5fold"] = avg_best_f1
    if best_epochs:
        avg_best_epoch = float(np.mean(best_epochs))
        print(f"Average best epoch across folds: {avg_best_epoch:.2f}")
        if wandb_run is not None:
            wandb.run.summary["avg_best_epoch"] = avg_best_epoch
    if best_overfit_gaps:
        avg_overfit_gap = float(np.mean(best_overfit_gaps))
        print(f"Average overfit gap (train_acc - val_acc): {avg_overfit_gap:.4f}")
        if wandb_run is not None:
            wandb.run.summary["avg_overfit_gap"] = avg_overfit_gap
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Multilevel (T1, T3-ctrl, T3-test) with CCT-LSTM")
    parser.add_argument('--data-root', type=str, default=None,
                        help="Path to '.../UBFC-Phys/Data'. If omitted, read from UBFC_data_path.txt")
    parser.add_argument('--stage1-weights-dir', type=str, default='weights',
        help="Directory containing Stage 1 best weights per fold")
    parser.add_argument('--output-dir', type=str, default='weights',
                        help="Directory to save Stage 2 best model weights")
    parser.add_argument('--report-dir', type=str, default='reports/levels',
                        help="Directory to save confusion matrices and reports")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stop-patience', type=int, default=8)

    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--freeze-cct', action='store_true', help='Freeze CCT backbones and train only LSTM+head')
    # CCT 正则参数
    parser.add_argument('--dropout', type=float, default=0.1, help='CCT dropout (default 0.1)')
    parser.add_argument('--emb-dropout', type=float, default=0.1, help='CCT embedding dropout (default 0.1)')

    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable logging to Weights & Biases (default: False).')
    parser.add_argument('--wandb-project', type=str, default='cct_lstm_levels',
                        help='wandb project name.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Optional wandb run name.')

    args = parser.parse_args()
    main(args)


