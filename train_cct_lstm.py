import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from preprocessing.dataset import VideoSequenceDataset, get_default_transforms
from model.model import CCT_LSTM_Model
from preprocessing.file_path_gen import FilePathGen


"""
Stage 2: Task classification (T1/T2/T3), 7-fold subject-level CV
----------------------------------------------------------------
This script trains the multimodal CCT-LSTM model on paired landmark/rPPG MTF
sequences. It expects Stage 1 best weights for both modalities, saved per-fold as:
  weights/cct_landmark_fold_{k}_best.pth
  weights/cct_rppg_fold_{k}_best.pth

Design choices
--------------
- Subject-level KFold (n_splits=7) to avoid subject leakage.
- Batch size = 1 because sequences can have different lengths across videos; this
  avoids padding logic and keeps code simple/robust.
- Optimizer: AdamW; Scheduler: ReduceLROnPlateau (mode=max on val_f1).
- Early stopping on val_f1 with patience.

Outputs
-------
- Best model per fold -> {output_dir}/cctlstm_tasks_fold_{k}_best.pth
- Confusion matrix (.npy) and classification report (.txt) under {report_dir}/fold_{k}
"""


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input has shape (B, T, 2, C, H, W). If shape is (T, 2, C, H, W), add B=1.
    """
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
        labels = labels.to(device, dtype=torch.long)

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
             device: torch.device) -> Tuple[Dict[str, float], np.ndarray, str]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for sequences, labels in tqdm(loader, desc="Validating", leave=False):
        sequences = _ensure_batch_dim(sequences).to(device)
        labels = labels.to(device, dtype=torch.long)

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
        data_root = Path(cli_data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        return data_root
    # fallback to UBFC_data_path.txt
    fpg = FilePathGen()
    return fpg.datapath


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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = get_data_root(args.data_root)
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('s')],
                      key=lambda x: int(x[1:]))
    if not subjects:
        raise RuntimeError(f"No subject folders found under {data_root}")

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

    # 7-fold subject-level CV
    kf = KFold(n_splits=7, shuffle=True, random_state=used_seed)

    # default CCT params aligned with Stage 1 for weight compatibility
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

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subjects), start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold_idx}/7")
        print("=" * 60)

        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")

        # Datasets / Loaders
        train_dataset = VideoSequenceDataset(data_root=data_root, subject_list=train_subjects, transform=transform)
        val_dataset = VideoSequenceDataset(data_root=data_root, subject_list=val_subjects, transform=transform)
        print(f"Sequences -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("[WARN] Empty dataset split detected; skipping this fold.")
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

        # Model
        model = CCT_LSTM_Model(cct_params=cct_params,
                                lstm_hidden_size=args.lstm_hidden_size,
                                lstm_num_layers=args.lstm_num_layers,
                                num_classes=3).to(device)

        # Load Stage 1 best weights for this fold
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
            # Put CCT into eval to disable dropout, keep LSTM/cls in train
            model.cct_landmark.eval()
            model.cct_rppg.eval()
            optim_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
        else:
            optim_params = model.parameters()

        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_path = output_dir / f"cctlstm_tasks_fold_{fold_idx}_best.pth"
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
                # save reports for the current best
                fold_dir = report_dir / f"fold_{fold_idx}"
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
            wandb.run.summary["avg_val_f1_7fold"] = avg_best_f1
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
    parser = argparse.ArgumentParser(description="Stage 2: Task classification (T1/T2/T3) with CCT-LSTM")
    parser.add_argument('--data-root', type=str, default=None,
                        help="Path to '.../UBFC-Phys/Data'. If omitted, read from UBFC_data_path.txt")
    parser.add_argument('--stage1-weights-dir', type=str, default='weights',
                        help="Directory containing Stage 1 best weights per fold")
    parser.add_argument('--output-dir', type=str, default='weights',
                        help="Directory to save Stage 2 best model weights")
    parser.add_argument('--report-dir', type=str, default='reports/tasks',
                        help="Directory to save confusion matrices and reports")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1, help="Kept for API symmetry; batch is fixed to 1")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stop-patience', type=int, default=40,
                        help="Early stopping patience on val_f1; 0 disables early stopping")

    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--freeze-cct', action='store_true', help='Freeze CCT backbones and train only LSTM+head')
    # CCT regularization parameters
    parser.add_argument('--dropout', type=float, default=0.1, help='CCT dropout (default 0.1)')
    parser.add_argument('--emb-dropout', type=float, default=0.1, help='CCT embedding dropout (default 0.1)')

    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable logging to Weights & Biases (default: False).')
    parser.add_argument('--wandb-project', type=str, default='cct_lstm',
                        help='wandb project name.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Optional wandb run name.')

    args = parser.parse_args()
    main(args)


