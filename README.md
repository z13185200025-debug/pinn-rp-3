# pinn-rp3

RP-3 航空煤油竖直圆管内流动换热与传热恶化预测框架。项目覆盖传统机器学习、ANN/DNN、CNN/U-Net、序列模型、DeepONet、FNO、GNN 以及 PINN/RANS-PINN/Property-PINN 的可扩展研究路径。

核心目标不是单纯追求最低误差，而是比较：

- prediction accuracy
- physical consistency
- cross-case generalization
- small-data robustness
- heat-transfer-deterioration interpretability

## 数据格式

原始 CFD 文件放在 `DATA/*.dat`，文件名规则：

```text
diameter-inlet_temperature-wall_heat_flux.dat
```

例如 `1.5-473-500.dat` 解析为 `diameter=1.5`、`inlet_temperature=473`、`wall_heat_flux=500`、`case_id=1.5-473-500`。

当前数据中 `y-coordinate` 基本接近 `1e-12`，真正径向坐标很可能是 `z-coordinate`。代码默认 `radial_coordinate: auto`，会在 `y/z/r/radial` 中按每个 case 的非零范围自动选择径向坐标，并写出 `data/processed/geometry_audit.parquet`。`r_over_R` 优先使用观测半径 `max(abs(radial))` 归一化，避免管径单位未确认时几何特征全错。

## 配置重点

- `configs/default.yaml`
  - `data_dir: DATA`
  - `radial_coordinate: auto`
  - `axial_coordinate: x`
  - `diameter_scale_to_meter: 1.0`
  - `heat_flux_scale_to_w_m2: 1000.0`
  - `grid_nx/grid_nr`
  - `sequence_length`
  - `num_workers/pin_memory`
- `configs/features.yaml`: 输入/输出特征、HTD 标签规则。
- `configs/model_config.yaml`: split、训练、各模型结构、PINN 损失权重。

如果 `geometry_audit` 显示 `observed_radius` 和 `diameter_physical/2` 偏差很大，需要人工确认管径单位。当前 smoke 数据显示偏差明显，因此代码使用观测半径构造 `r_over_R`。

## 数据构建产物

运行构建后会写入 `data/processed/`：

- `full_field_dataset.parquet`: 网格点级数据。
- `wall_dataset.parquet`: 壁面/近壁沿程数据。
- `case_summary_dataset.parquet`: 每 case 一行的风险摘要。
- `geometry_audit.parquet`: 径向坐标选择和管径单位审计。
- `grid_dataset.npz` + `grid_metadata.json`: CNN/U-Net/FNO 规则 `x-r` 网格。
- `wall_sequence_dataset.npz` + `wall_sequence_metadata.json`: LSTM/GRU/TCN/Transformer 序列数据。

`data/*.dat` 即使存在也不会被当作 processed 数据；脚本默认只从 `DATA/` 读原始数据，只向 `data/processed/` 写构建结果。

## 安装

```bash
pip install -r requirements.txt
```

基础依赖：pandas、numpy、scikit-learn、matplotlib、seaborn、PyTorch、PyYAML、tqdm、pyarrow、scipy、tabulate。

可选依赖：

- `xgboost`
- `lightgbm`
- `torch_geometric`，仅 GNN/MeshGraphNet 需要。缺失时 CLI 会清晰跳过，不影响其他模型。

## 最小 Smoke Workflow

```bash
python -m src.scripts.build_dataset --data_dir DATA --config configs/default.yaml --features_config configs/features.yaml --max_cases 4
python -m src.scripts.run_eda
python -m src.scripts.train_baselines --model mlp --sample_rows 20000
python -m src.scripts.train_baselines --model resnet_mlp --sample_rows 20000
python -m src.scripts.train_pinn --sample_rows 5000
python -m src.scripts.compare_models --metrics_dir outputs/metrics
```

## 单独训练某个模型

表格型 baseline 和基础 PINN 仍保留专用入口：

```bash
python -m src.scripts.train_baselines --model mlp --sample_rows 20000 --epochs 5
python -m src.scripts.train_baselines --model resnet_mlp --sample_rows 20000 --epochs 5
python -m src.scripts.train_baselines --model random_forest --sample_rows 20000
python -m src.scripts.train_pinn --sample_rows 5000 --epochs 5
```

grid、sequence 和 operator 类模型使用统一入口：

```bash
python -m src.scripts.train_model --model cnn --task grid --smoke
python -m src.scripts.train_model --model unet --task grid --smoke
python -m src.scripts.train_model --model fno --task grid --smoke
python -m src.scripts.train_model --model lstm --task sequence --smoke
python -m src.scripts.train_model --model transformer --task sequence --smoke
python -m src.scripts.train_model --model deeponet --task field --sample_rows 10000 --epochs 2
python -m src.scripts.train_model --model property_pinn --task field --sample_rows 10000 --epochs 2
python -m src.scripts.train_model --model rans_pinn --task field --sample_rows 10000 --epochs 2
```

统一入口支持：

```text
--model mlp|dnn|resnet_mlp|cnn|unet|fno|lstm|gru|tcn|transformer|deeponet|property_pinn|rans_pinn|gnn
--task field|grid|sequence|graph
```

也可以用 `src.main`：

```bash
python -m src.main train_model --model fno --task grid --smoke
```

## 一起训练多个或所有模型

推荐先用 smoke/sampled 模式批量训练，确认环境和数据产物都正常：

```bash
python -m src.scripts.train_all --models all --smoke --epochs 2
```

只训练一组选定模型：

```bash
python -m src.scripts.train_all --models mlp,resnet_mlp,pinn,cnn,fno,lstm,deeponet --smoke --epochs 2
```

用较大样本训练所有默认模型：

```bash
python -m src.scripts.train_all --models all --sample_rows 200000 --pinn_sample_rows 50000 --epochs 20
```

`train_all` 会顺序调用各模型训练脚本。`gnn` 如果缺少 `torch_geometric` 会输出 skip 信息；`xgboost/lightgbm` 如果未安装，需要先安装依赖或从 `--models` 中去掉。

训练结束后汇总对比表：

```bash
python -m src.scripts.compare_models --metrics_dir outputs/metrics
```

## 模型状态

已可训练或 smoke-run：

- RandomForest / SVR / optional XGBoost / optional LightGBM
- MLP / DNN，支持 activation、dropout、BatchNorm/LayerNorm、residual、Fourier features
- ResNet-MLP
- CNNRegressor / U-Net，支持 mask loss
- FNO2d，多层 spectral conv + pointwise conv
- LSTM / GRU / TCN / Transformer，支持 padded sequence mask
- DeepONet，branch 输入 `[diameter, inlet_temperature, wall_heat_flux]`，trunk 输入 `[x, r]`
- PropertyPINN / PropertyNet，缺少真实物性表时从 CFD 学习 `T/p/case params -> rho/Pr`
- RANSPINN，可配置辅助输出接口
- GNN/MeshGraphNet，安装 `torch_geometric` 后可用；否则跳过

## Split 策略

默认 `case_split`，保证同一 `case_id` 不会同时出现在训练集和测试集。还支持：

- `point_random_split`
- `leave_one_diameter_out`
- `leave_one_heat_flux_out`
- `leave_one_inlet_temperature_out`

split manifest 保存到 `outputs/reports/split_manifest.json`。所有 scaler 只在 train split 上 fit，并保存到对应模型目录，例如 `outputs/models/mlp/`，避免 test leakage。

## PINN 物理损失

基础 PINN 输入为：

```text
[diameter, inlet_temperature, wall_heat_flux, x, x_over_L, r_over_R]
```

输出使用数据中实际存在的目标列，优先：

```text
[temperature, u, pressure, density]
```

如果 pressure 缺失，会自动降级，不会崩。

PINN loss 当前包括：

- supervised data loss
- weak continuity: `d(rho*u)/dx`
- simplified axisymmetric energy:
  `rho*Cp*u*dT/dx - k*(d2T/dx2 + d2T/dr2 + safe(1/r)*dT/dr)`
- inlet temperature BC
- wall heat flux BC: `-k*dT/dr = q_wall`
- wall no-slip BC

网络仍使用 scaled input/output 训练，但物理残差先反标准化，并用 scaler 标准差做链式导数缩放。近轴 `1/r` 使用稳定下界避免 NaN/Inf。

限制：当前仍是简化二维轴对称能量约束；完整论文级模型应继续扩展到 RANS、湍流闭合、变物性和真实 RP-3 物性表。

## 输出

- `outputs/models/<model_name>/`: checkpoints、scalers、metadata。
- `outputs/metrics/<model_name>/`: JSON metrics。
- `outputs/predictions/<model_name>/`: prediction CSV，包含 `case_id/x/r_over_R` 等定位列。
- `outputs/figures/<model_name>/`: loss、parity、error distribution、feature importance。
- `outputs/figures/eda/`: EDA 图。
- `outputs/reports/model_comparison.csv`
- `outputs/reports/model_comparison.md`

例如 MLP 的输出会写入：

```text
outputs/models/mlp/
outputs/metrics/mlp/mlp_metrics.json
outputs/predictions/mlp/mlp_predictions.csv
outputs/figures/mlp/
```
## 需确认

- 文件名中的 `diameter` 单位。
- `wall_heat_flux` 是否为 kW/m2；当前默认乘 `1000.0` 转 W/m2。
- CFD 坐标单位以及 `z-coordinate` 是否确认为径向坐标。
- 是否有 RP-3 真实物性表或经验关联式。
- 是否需要完整 RANS 方程、湍流模型和变物性闭合。
# pinn-rp-3
