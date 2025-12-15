# CCT-LSTM Reproduce（研究版 README）

本说明面向需要深入理解 CCT-LSTM 复现管线的研究者，重点覆盖目录组织、两阶段训练关系、张量流向以及 Stage 1 训练脚本的核心参数。

## 项目结构

当前仓库的关键目录如下（截取至二级）：

```
CCT-LSTM_reproduce
├── train_cct.py                # Stage 1：单模态 CCT 预训练
├── train_cct_lstm_levels.py    # Stage 2：多模态 CCT-LSTM 微调
├── train_cct_lstm.py           # （可选）多任务/多标签版本
├── model/
│   └── model.py                # CCT & CCT-LSTM 定义
├── preprocessing/
│   ├── dataset.py              # Stage1/Stage2 数据加载器
│   └── landmark_extractor.py   # 数据生成相关脚本
├── utils/
│   └── summarize_reports.py    # 实验报告整理
├── weights/                    # 各折最优权重缓存
├── reports/                    # 混淆矩阵与 CSV 汇总
└── logs/, wandb/               # 训练日志与可视化
```

## 训练工作流（Stage 1 → Stage 2）

- **Stage 1（`train_cct.py`）— 模态特定的 CCT 预训练**
  - 输入：单帧 MTF 图像（landmark 或 rPPG）。
  - 目标：以 7 折交叉验证方式对每个模态训练一个 `CCTForPreTraining` 分类器，输出张量形状为 `[B, 3]`，并保存每折最优权重 `cct_<modality>_fold_k_best.pth`。
  - 技巧：可选的 Mixup、Cutout、噪声增强提升泛化，并使用 CosineAnnealingLR 和早停策略。

- **Stage 2（`train_cct_lstm_levels.py`）— 多模态序列微调**
  - 输入：按视频窗口拼接的多帧序列，形状 `[B, T, 2, C, H, W]`，`2` 对应 landmark/rPPG。
  - 操作：加载 Stage 1 对应折的权重初始化两个 CCT 分支（可选择冻结），再拼接特征送入 LSTM 捕获时序依赖，最终通过线性分类头完成多级压力分类（T1、T3-ctrl、T3-test）。
  - 评估：5 折分层交叉验证，保存 `cctlstm_levels_fold_k_best.pth` 及混淆矩阵/报告，调度器采用 ReduceLROnPlateau。

因此，Stage 1 负责建立模态感知的空间表征，Stage 2 在其基础上融合多模态并建模时间序列，实现完整的 CCT-LSTM 推理链路。

## 数据与张量流向

### Stage 1：单帧样本

`SingleImageDataset` 会遍历 `mtf_images/<modality>/` 下的窗口图像，并将每张图片转换为 `[C, H, W] = [3, 224, 224]` 的归一化张量，Label 取自文件名中的 T1/T2/T3。该张量直接输入 `CCTForPreTraining`，模型输出 3 维 logits 供交叉熵训练。

```217:225:preprocessing/dataset.py
        landmark_sequence = torch.stack(landmark_tensors)
        rppg_sequence = torch.stack(rppg_tensors)
        final_sequence = torch.stack([landmark_sequence, rppg_sequence], dim=1)
        return final_sequence, label
```

### Stage 2：序列样本与多模态融合

`VideoSequenceDataset`/`LevelsSequenceDataset` 将同一 subject、同一 level 的 landmark/rPPG 窗口对齐成序列（长度 ≥10）。堆叠后形成 `final_sequence`，其 batch 版本进入 `CCT_LSTM_Model`：

1. 分离两种模态：`[B, T, 2, C, H, W] → [B·T, C, H, W]`。
2. 分别通过共享结构的 CCT 骨干，得到 `[B·T, d]` 嵌入。
3. 拼接为 `[B, T, 2d]`，送入 `nn.LSTM(batch_first=True)`。
4. 取最后一层隐藏状态 `[B, hidden]`，经线性层输出 `[B, 3]`。

```165:204:model/model.py
        landmark_seq = x[:, :, 0, :, :, :]
        rppg_seq = x[:, :, 1, :, :, :]
        landmark_flat = landmark_seq.reshape(batch_size * seq_len, c, h, w)
        rppg_flat = rppg_seq.reshape(batch_size * seq_len, c, h, w)
        landmark_features = self.cct_landmark(landmark_flat)
        rppg_features = self.cct_rppg(rppg_flat)
        combined_features = torch.cat([landmark_features, rppg_features], dim=1)
        feature_sequence = combined_features.view(batch_size, seq_len, -1)
        _, (h_n, _) = self.lstm(feature_sequence)
        output = self.classifier(h_n[-1])
```

该设计确保每帧先由 CCT 提取局部空间模式，再由 LSTM 捕捉跨窗口的动态变化（如血容积波动节律），满足多模态时序融合需求。

## `train_cct.py` 关键参数

| 参数 | 默认值 | 作用说明 |
| --- | --- | --- |
| `--modality {landmark,rppg}` | 必填 | 指定当前预训练的模态，控制数据目录与权重命名。 |
| `--data-root PATH` | `None` | 数据根目录；缺省则读取 `UBFC_data_path.txt`。 |
| `--output-dir PATH` | `weights` | 保存每折最佳权重的目录。 |
| `--epochs` | `100` | 每折最大训练轮数。 |
| `--batch-size` | `16` | DataLoader 的 batch 大小。 |
| `--lr` | `1e-4` | AdamW 学习率。 |
| `--wd` | `1e-4` | AdamW 权重衰减，配合 dropout 抑制过拟合。 |
| `--seed` | `-1` | 负值表示随机采样种子并打印，正值则复现。 |
| `--device {cuda,cpu}` | `cuda` | 训练所用设备；如 GPU 不可用自动回退 CPU。 |
| `--num-workers` | `4` | DataLoader 并行加载线程数。 |
| `--early-stop-patience` | `30` | 验证 F1 无提升时的提前停止耐心值；0 关闭。 |
| `--dropout` / `--emb-dropout` | `0.1` | CCT 主干与嵌入层的 dropout 概率。 |
| `--use-mixup` + `--mixup-alpha` | `False` / `0.2` | 是否启用 Mixup 及其 Beta 分布参数。 |
| `--use-cutout` + `--cutout-n-holes` + `--cutout-length` | `False` / `2` / `40` | 控制随机遮挡增强的孔洞数量与尺寸。 |
| `--use-noise` + `--noise-std` | `False` / `0.03` | 是否在 ToTensor 后加入高斯噪声及其标准差。 |
| `--use-wandb` | `False`（脚本默认值 True, 需显式开启） | 是否上传指标到 Weights & Biases。 |
| `--wandb-project` / `--wandb-run-name` | `cct_pretraining` / `None` | 实验跟踪的项目名与可选 run 名称。 |

组合以上参数即可灵活配置不同模态的预训练策略，并为 Stage 2 提供高质量初始化。

---

如需在不影响原始 `README.md` 的情况下分发本研究文档，可直接引用本文件（`README_research.md`）或在合并分支时再行替换。欢迎根据实验需求补充更多结果与可复现指令。***

