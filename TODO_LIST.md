# TODO 列表（Tech Debt）

## Hardcoded Paths / Values
- [ ] 将 `preprocessing/rppg_extractor.py` 中的 `DATASET_ROOT_PATH = Path("/mnt/f/DataSet/UBFC-Phys/Data")`、`WINDOW_LENGTH_SEC = 60` 等常量改为 CLI/配置输入，避免在非作者机器上直接报错。
- [ ] 让 `preprocessing/file_path_gen.py` 真正尊重传入的 `config_path`，不要再次硬编码 `'UBFC_data_path.txt'`，确保多数据根或多环境可用。
- [ ] 为 Stage 2 CCT-LSTM 的类别数提供 CLI/配置（目前在 `train_cct_lstm.py` 与 `train_cct_lstm_levels.py` 中直接传 `num_classes=3`），同时让 `model/CCT_LSTM_Model` 默认值改为依赖外部参数。
- [ ] 将 `preprocessing/dataset.py` 中的 label_map（`{'T1':0,'T2':1,'T3':2}`）与 `MIN_SEQ_LENGTH=10` 参数化；现在若扩展任务，只能手改源码。

## Dead Code / 未整合模块
- [ ] 处理 `utils/summarize_reports.py`：当前既不被训练脚本调用，又将 `reports_root` 解析为 `utils/reports`（路径错误），需决定是整合到评估流程还是改写/迁移到 docs。
- [ ] 明确 `preprocessing/rppg_extractor.py` 的职责——该文件依赖 pyVHR 且不在主流程中 import；若仅为参考，应搬到 `docs/` 或提供独立入口，否则应创建受支持的执行路径与依赖声明。

## Redundant / Unused Arguments
- [ ] `train_cct_lstm.py` 定义了 `--batch-size` 却始终将 DataLoader 固定为 1；需要删除该参数或真正将其接入 loader 构建逻辑，保持 CLI 一致性。


