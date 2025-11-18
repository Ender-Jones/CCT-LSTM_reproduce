import argparse
import os
import random
from pathlib import Path
import sys
from typing import Dict, Any, List
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
 

from dataset import SingleImageDataset, get_default_transforms
from model import CCTForPreTraining
from file_path_gen import FilePathGen


def set_seed(seed: int):
    """
    设置随机种子以确保实验的可复现性。

    Args:
        seed (int): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Data augmentations for MTF (train-only) ---
class RandomCutout(object):
    def __init__(self, n_holes: int = 1, length: int = 32):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W), assumed in [0,1] right after ToTensor, before Normalize
        c, h, w = img.shape
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask_t = torch.from_numpy(mask).to(img.dtype).to(img.device)
        mask_t = mask_t.expand_as(img)
        return img * mask_t


class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 0.03):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W), assumed in [0,1] right after ToTensor, before Normalize
        return img + torch.randn_like(img) * self.std + self.mean


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device: torch.device = torch.device('cuda')):
    lam = np.random.beta(alpha, alpha) if alpha and alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    use_mixup: bool = False, mixup_alpha: float = 0.4) -> Dict[str, float]:
    """
    对模型进行一个 epoch 的训练。

    Args:
        model (nn.Module): 待训练的模型。
        loader (DataLoader): 训练数据的 DataLoader。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 计算设备 (cuda/cpu)。
        use_mixup (bool): 是否启用 Mixup。
        mixup_alpha (float): Mixup 的 Beta 分布参数。

    Returns:
        Dict[str, float]: 包含平均训练损失、准确率和 F1 分数的字典。
    """
    model.train()  # 将模型设置为训练模式

    # TODO: 初始化用于累计损失和评估指标的变量
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        # TODO: 将数据移动到指定设备
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        # --- 学习四步曲 ---
        # 1. 梯度清零
        optimizer.zero_grad(set_to_none=True)
        # 2. 前向传播 (+ Mixup)
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            predictions = model(images)
            # 3. 计算损失 (Mixup)
            loss = mixup_criterion(criterion, predictions, targets_a, targets_b, lam)
            current_labels = targets_a if lam >= 0.5 else targets_b
        else:
            predictions = model(images)
            # 3. 计算损失
            loss = criterion(predictions, labels)
            current_labels = labels
        # 4. 反向传播与权重更新
        loss.backward()
        optimizer.step()

        # TODO: 累计损失和预测结果/真实标签，用于后续指标计算
        total_loss += loss.item()
        preds = torch.argmax(predictions, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(current_labels.detach().cpu().numpy())

    # TODO: 计算整个 epoch 的平均损失和各项指标 (accuracy, f1-score)
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds, normalize=True)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {'train_loss': avg_loss, 'train_acc': accuracy, 'train_f1': f1}


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    在验证集上评估模型性能。

    Args:
        model (nn.Module): 待评估的模型。
        loader (DataLoader): 验证数据的 DataLoader。
        criterion (nn.Module): 损失函数。
        device (torch.device): 计算设备 (cuda/cpu)。

    Returns:
        Dict[str, float]: 包含平均验证损失、准确率和 F1 分数的字典。
    """
    model.eval()  # 将模型设置为评估模式

    # TODO: 初始化用于累计损失和评估指标的变量
    total_loss = 0.0
    all_preds, all_labels = [], []

    # 在 no_grad 环境下进行，以节省计算资源
    with torch.no_grad():
        for images, labels in loader:
            # TODO: 将数据移动到指定设备
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            # TODO: 前向传播
            predictions = model(images)

            # TODO: 计算损失
            loss = criterion(predictions, labels)

            # TODO: 累计损失和预测结果/真实标签
            total_loss += loss.item()
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # TODO: 计算整个验证集的平均损失和各项指标
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds, normalize=True)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {'val_loss': avg_loss, 'val_acc': accuracy, 'val_f1': f1}


def main(args: argparse.Namespace):
    """
    主执行函数，负责组织整个交叉验证、训练和验证流程。

    Args:
        args (argparse.Namespace): 从命令行解析的参数。
    """
    # 1. 设置与初始化（支持随机种子并打印）
    def _init_and_set_seed(seed_arg: int) -> int:
        if seed_arg is None or seed_arg < 0:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(seed_arg)
        print(f"Using seed: {seed}")
        set_seed(seed)
        return seed

    used_seed = _init_and_set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Weights will be saved to: {output_dir}")

    # 可选：初始化 Weights & Biases 进行实验跟踪
    wandb_run = None
    if getattr(args, "use_wandb", False):
        wandb_config = vars(args).copy()
        wandb_config["used_seed"] = used_seed
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    # data root and all subjects
    if args.data_root is not None:
        data_root = Path(args.data_root)
        if data_root.exists() and data_root.is_dir():
            all_subjects = sorted(
                [d.name for d in data_root.iterdir() if d.is_dir()],
                key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else x
            )
        else:
            print(f"Data root: {data_root} not found. Please specify a valid data root.")
            sys.exit(1)
    else:
        with open("UBFC_data_path.txt", "r") as f:
            data_root = Path(f.read().strip())
        if data_root.exists() and data_root.is_dir():
            all_subjects = sorted(
                [d.name for d in data_root.iterdir() if d.is_dir()],
                key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else x
            )
        else:
            print(f"Data root: {data_root} not found. Please specify a valid data root.")
            sys.exit(1)

    kf = KFold(n_splits=7, shuffle=True, random_state=used_seed)

    # TODO: 初始化一个列表，用于存储每个 fold 的最佳结果，以便最后计算平均值
    # 然而, 我并不确定论文原文的内容是否是这么实现的, 需要检查笔记确认一下
    all_folds_best_metrics = []
    best_epochs: List[int] = []
    best_overfit_gaps: List[float] = []

    # 3. 交叉验证主循环
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_subjects)):
        print("\n" + "=" * 50)
        print(f"Cross-Validation Fold {fold_idx + 1}/7")
        print("=" * 50)

        # a. 划分当前 fold 的训练集和验证集
        train_subjects = [all_subjects[i] for i in train_indices]
        validation_subjects = [all_subjects[i] for i in val_indices]
        print(f"Training on {len(train_subjects)} subjects.")
        print(f"Validating on {len(validation_subjects)} subjects.")
        if args.early_stop_patience > 0:
            print(f"Early stopping enabled with patience={args.early_stop_patience} (monitor=val_f1)")

        # b. 创建数据集和 DataLoader
        # TODO: 使用 train_subjects 和 validation_subjects 分别实例化 SingleImageDataset
        # 验证集保持干净
        val_transform = get_default_transforms()

        # 训练集：Resize -> ToTensor -> (Noise) -> (Cutout) -> Normalize
        train_tf_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        if args.use_noise:
            train_tf_list.append(AddGaussianNoise(std=args.noise_std))
        if args.use_cutout:
            train_tf_list.append(RandomCutout(n_holes=args.cutout_n_holes, length=args.cutout_length))
        train_tf_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
        train_transform = transforms.Compose(train_tf_list)

        train_dataset = SingleImageDataset(
            data_root=data_root,
            modality=args.modality,
            subject_list=train_subjects,
            transform=train_transform
        )

        val_dataset = SingleImageDataset(
            data_root=data_root,
            modality=args.modality,
            subject_list=validation_subjects,
            transform=val_transform
        )

        # TODO: 创建 DataLoader
        train_dataLoader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=(args.num_workers > 0))

        val_dataLoader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=(args.num_workers > 0))

        # c. 为当前 fold 创建全新的模型、优化器和损失函数
        # TODO: 实例化模型并移动到 device
        model = CCTForPreTraining(
            num_classes=3,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout
        ).to(device)
        # TODO: 实例化优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # d. 训练与验证循环
        # TODO: 初始化用于追踪当前 fold 最佳性能的变量
        best_val_f1 = 0.0
        epochs_no_improve = 0
        best_epoch_for_fold = None
        best_overfit_gap_for_fold = None

        epoch_times: List[float] = []
        for epoch in range(args.epochs):
            start_t = time.time()

            train_metrics = train_one_epoch(
                model, train_dataLoader, criterion, optimizer, device,
                use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
            )
            val_metrics = validate(model, val_dataLoader, criterion, device)
            scheduler.step()

            dur = time.time() - start_t
            epoch_times.append(dur)
            avg_t = sum(epoch_times) / len(epoch_times)
            remain = (args.epochs - (epoch + 1)) * avg_t

            print(
                f"[Fold {fold_idx + 1}] Epoch {epoch + 1}/{args.epochs} | "
                f"Train loss={train_metrics['train_loss']:.4f} acc={train_metrics['train_acc']:.4f} | "
                f"Val loss={val_metrics['val_loss']:.4f} acc={val_metrics['val_acc']:.4f} f1={val_metrics['val_f1']:.4f} | "
                f"time={dur:.2f}s ETA~{remain/60:.1f}m"
            )

            # 将当前 epoch 的指标发送到 wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["train_loss"],
                        "train_acc": train_metrics["train_acc"],
                        "train_f1": train_metrics["train_f1"],
                        "val_loss": val_metrics["val_loss"],
                        "val_acc": val_metrics["val_acc"],
                        "val_f1": val_metrics["val_f1"],
                        "overfit_gap": train_metrics["train_acc"] - val_metrics["val_acc"],
                        "epoch_time_sec": dur,
                    }
                )

            # e. 检查并保存最佳模型
            current_f1 = val_metrics['val_f1']
            if current_f1 > best_val_f1 + 1e-6:
                best_val_f1 = current_f1
                epochs_no_improve = 0
                save_path = output_dir / f"cct_{args.modality}_fold_{fold_idx + 1}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path} with F1 score: {best_val_f1:.4f}")

                best_epoch_for_fold = epoch + 1
                best_overfit_gap_for_fold = train_metrics["train_acc"] - val_metrics["val_acc"]

                # 在 wandb summary 中记录当前 fold 的最好 F1
                if wandb_run is not None:
                    wandb.run.summary[f"best_val_f1_fold_{fold_idx + 1}"] = best_val_f1
                    wandb.run.summary[f"best_epoch_fold_{fold_idx + 1}"] = best_epoch_for_fold
                    wandb.run.summary[f"overfit_gap_fold_{fold_idx + 1}"] = best_overfit_gap_for_fold
            else:
                epochs_no_improve += 1

            # f. 提前停止判断
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"[Fold {fold_idx + 1}] Early stopping at epoch {epoch + 1} "
                      f"(no val_f1 improvement for {epochs_no_improve} epochs)")
                break

        # TODO: 记录当前 fold 的最佳 F1 分数
        all_folds_best_metrics.append({'fold': fold_idx + 1, 'best_f1': best_val_f1})
        if best_epoch_for_fold is not None:
            best_epochs.append(best_epoch_for_fold)
        if best_overfit_gap_for_fold is not None:
            best_overfit_gaps.append(best_overfit_gap_for_fold)
        print(f"\nBest F1 score for fold {fold_idx + 1}: {best_val_f1:.4f}")

    # 4. 总结并打印所有 fold 的平均性能
    # TODO: 计算并打印交叉验证的平均 F1 分数
    avg_f1 = np.mean([m["best_f1"] for m in all_folds_best_metrics])
    avg_best_epoch = float(np.mean(best_epochs)) if best_epochs else None
    avg_overfit_gap = float(np.mean(best_overfit_gaps)) if best_overfit_gaps else None
    print("\n" + "=" * 50)
    print("Cross-Validation Finished!")
    print(f"Average F1 score across 7 folds: {avg_f1:.4f}")
    if avg_best_epoch is not None:
        print(f"Average best epoch across folds: {avg_best_epoch:.2f}")
    if avg_overfit_gap is not None:
        print(f"Average overfit gap (train_acc - val_acc): {avg_overfit_gap:.4f}")
    print("=" * 50)

    # 在 wandb summary 中记录整体 7-fold 平均
    if wandb_run is not None:
        wandb.run.summary["avg_val_f1_7fold"] = float(avg_f1)
        if avg_best_epoch is not None:
            wandb.run.summary["avg_best_epoch"] = avg_best_epoch
        if avg_overfit_gap is not None:
            wandb.run.summary["avg_overfit_gap"] = avg_overfit_gap
        wandb_run.finish()


if __name__ == '__main__':
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="Stage 1: CCT Pre-training Script")

    # 核心参数
    parser.add_argument('--modality', type=str, required=True, choices=['landmark', 'rppg'],
                        help="The modality to train on ('landmark' or 'rppg').")
    parser.add_argument('--data-root', type=str, default=None,
                        help="Path to the root of the dataset (e.g., '.../UBFC-Phys/Data'). "
                             "If not provided, will try to read from 'UBFC_data_path.txt'.")
    parser.add_argument('--output-dir', type=str, default='weights',
                        help="Directory to save the best model weights.")

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size for training and validation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for the optimizer. Default is 1e-4.")
    parser.add_argument('--wd', type=float, default=1e-4,
                        help="Weight decay for the optimizer. Default is 1e-4.")
    # 环境与复现性参数
    parser.add_argument('--seed', type=int, default=-1,
                        help="Random seed. <0 to sample a new random seed per run and print it.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use for training ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--early-stop-patience', type=int, default=30,
                        help='Early stopping patience on val_f1; 0 表示关闭（默认 30）')
    # CCT 正则参数
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='CCT dropout rate (default: 0.1)')
    parser.add_argument('--emb-dropout', type=float, default=0.1,
                        help='CCT embedding dropout rate (default: 0.1)')
    # 训练数据增强（默认关闭）
    parser.add_argument('--use-mixup', action='store_true',
                        help='Enable Mixup for training (default: False).')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Alpha parameter for Beta distribution in Mixup (default: 0.4).')
    parser.add_argument('--use-cutout', action='store_true',
                        help='Enable Random Cutout for training (default: False).')
    parser.add_argument('--cutout-n-holes', type=int, default=2,
                        help='Number of holes for Cutout (default: 2).')
    parser.add_argument('--cutout-length', type=int, default=40,
                        help='Side length of each hole for Cutout (default: 40).')
    parser.add_argument('--use-noise', action='store_true',
                        help='Enable Gaussian noise for training (default: False).')
    parser.add_argument('--noise-std', type=float, default=0.03,
                        help='Std of Gaussian noise added after ToTensor and before Normalize (default: 0.03).')

    # Weights & Biases 实验追踪
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable logging to Weights & Biases (default: True).')
    parser.add_argument('--wandb-project', type=str, default='cct_pretraining',
                        help='wandb project name.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Optional wandb run name.')

    args = parser.parse_args()

    main(args)
