# statementDm — DReSTGy → DReSTG-main 匿名复现版重构与实验生命周期合同

> 版本：2026-11-18。唯一修改基线：用户上传的 `DReSTGy.zip`；输出目录根 `DReSTG-main/`，其下直接为 `configs/`、`datasets/`、`data_provider/`、`exp/`、`layers/`、`models/`、`scripts/`、`utils/`、`run.py` 等文件和目录。`TVC05_reference和prefix路径详细.zip` 仅用于 Reference 机制对照，不复制其文件/旧 schema。此 statement 是实施前冻结约束、逐文件改造台账与实施后验收记录；单独提供 `statementDm.md`，并置入新仓库根目录。

## 一、裁决、范围、必须保持的东西

1. **科学计算图完全冻结**：全历史 RevIN → 不重新从粗窗口起点切分的整段 50% overlap Fine `unfold` →共享 Fine 线性嵌入 → 包含跨粗窗口边界 Fine Token 的 Coarse 均值池化 → Fine/Coarse interleave 与位置编码 → 第一层残差时序 Attention+FFN → MeanPool(`N_C=7`) Global Nodes → `D_R=D/4` learned projected cosine → positive cumulative-mass Soft-K → ST Pearson positive cumulative-mass Soft-K → 通道对 Union → Global 正权 MH1 与零边 MH1 的差 → `softplus(theta_lambda)` 乘增量并广播到 Coarse Nodes → 由 `atanh(rho)+softplus(theta_s)*atanh(stat cosine)` 经 `tanh`、`ReLU` 构成 ST 边权 → ST MH1 → 剩余时序/图层 → 仅 Fine Tokens 做 `D × N_F` 维先尾拼接展平 → Linear → inverse RevIN。禁止修改任何一个运算、维度、张量排列、初始化顺序、随机数调用顺序、前向循环、训练/测试计算或数值防护常量。
2. **Reference 计算机制冻结**：`base/<dataset>/<train|val|test>/{rho.npy,stat.npy,idx.npy,meta.json,ok.json}`；`st/<dataset>/<split>/{off.npy,pair.npy,r.npy,meta.json,ok.json}`；full-history signed Pearson；原始数据每 coarse 区间 mean/std 后非仿射 LayerNorm；canonical `c<d`；正 Pearson 严格 `>0`；raw-positive cumulative-mass Soft-K（`K_s=floor(k_m*C(C-1)/2)`，浮点双精度累加、固定 tie-break、不 refill）；packed/ragged、mmap；先建临时目录，exact verify，`VERIFIED` 后原子发布；`reuse` 只验证/读取本项目 root，不进行跨项目 fallback。**只重命名 Reference schema / 术语和 CLI 标签**，保持 `base/st` 物理路径、文件布局、数值算法和数据接口。
3. **实验参数不可调整**：七数据集 × 4 horizon = 28 cells；历史长度 336；horizon 96/192/336/720；单 seed 2021；epochs=100；Trace/分割/训练/验证/测试方法原样；Traffic batch 8；其他 batch、学习率、调度、dropout、图预算、模型宽度等均逐字继承 YAML。特别冻结 `mixhop_depth=1`、`mixhop_alpha=0.05`、`theta_s_init=0.5413248546`（旧 `theta_stat_init` 同数值）、`boundary_eps=0.001`、`lambda_init=0.1`；`affine=False`、`decomposition=False`、`subtract_last=False`。
4. **接口/目录语义对齐论文**：`Global Nodes H^G`、`Coarse Patch Nodes H^C`、`Fine Patch Tokens H^F`；全局图 `G`，时空图 `ST`；`Local` 仅表示边/图的局部关系语义；使用 `N_C/P_C/P_F/S_F/N_F` 语义变量；不再把旧 `patch_num=N_C*M_F` 冒称全局 Fine token 数 `N_F`。Ref cache 参数 `l/patch_len/stride` 仍保留作为存储合同内部字段（其数值与来源同步对齐），其物理含义不变。
5. **公开命名脱敏且保留版权来源**：删除旧阶段 statement、重命名主路径出现的 TVC/TVC3/TVC5/DReSTGx/DReSTGy/R2/O1 等身份文字；源文件已有 Yuan Feng / Nie / PatchTST 等版权声明和 Apache-2.0/MIT 来源保留，可将注释中的“改自某项目”作为准确出处保留，禁止伪造完全独立开发的来源。完整 LICENSE 不动。
6. **接口变动决定历史资产边界**：此次重命名会改变 torch `state_dict` keys；新仓库从头训练且不保证无迁移加载旧 checkpoint。Reference 新 schema 无法直接读取旧 `drestgx_*` `meta.json/ok.json`；使用 `DRESTG_REFERENCE_MODE=build` 重建（数值定义未变）；既有旧资产需另行 metadata migration，禁止直接谎报兼容。原 DReSTGy 文件保持未修改，可用于恢复/复核。

## 二、冻结的实验配置（均继承 `DReSTGy.zip/configs/*.yaml`）

| Dataset | C | N_E | D | N_A | D_FF | P_F/S_F | Batch | LR | Dropout | Patience | LR policy | k_m |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| ETTh1 | 7 | 2 | 32 | 8 | 128 | 12/6 | 128 | 0.001 | 0.5 | 5 | type3, pct=0.3 | 0.50 |
| ETTh2 | 7 | 2 | 32 | 8 | 128 | 12/6 | 128 | 0.0001 | 0.5 | 10 | type3, pct=0.3 | 0.25 |
| ETTm1 | 7 | 2 | 32 | 8 | 128 | 12/6 | 128 | 0.001 | 0.5 | 10 | type3, pct=0.3 | 0.35 |
| ETTm2 | 7 | 2 | 32 | 16 | 128 | 12/6 | 128 | 0.0001 | 0.5 | 10 | type3, pct=0.3 | 0.65 |
| Weather | 21 | 2 | 64 | 16 | 256 | 24/12 | 128 | 0.001 | 0.2 | 5 | 原 YAML 默认 type3, pct=0.3 | 0.20 |
| Electricity | 321 | 3 | 64 | 16 | 256 | 12/6 | 16 | 0.0005 | 0.2 | 5 | 原 YAML 默认 type3, pct=0.3 | 0.01 |
| Traffic | 862 | 3 | 128 | 16 | 256 | 16/8 | 8 | 0.0001 | 0.2 | 10 | TST, pct=0.2 | 0.0037 |

*所有配置 `T=336,N_C=7,P_C=48,train_epochs=100,itr=1,d_layers=1,gnn_r_degree=1,mixhop_depth=1,mixhop_alpha=0.05,mixhop_support_preserving=False,affine=False`。ETT `gnn_hidden_size=32`；Weather=64、Electricity=32；Traffic YAML 原本没有显式 gnn_hidden_size，继续沿用入口默认 32。保留 `graph_conv_hidden_size/graph_conv_kernel_size/subgraph_keep_rate` 的历史 YAML 值：它们目前不参与本计算分支，第一阶段不删除。ETTm2 的 `n_heads=16` 原样保留。Weather/Electricity 省略的 head/fc dropout、调度等由原入口默认值决定，不得隐式改成他值。*

## 三、逐文件修改台账（位置 → 内容 → 原因）

| 位置 | 本次内容 | 理由与等价性边界 |
|---|---|---|
| `run_TVCFormer.py` → `run.py` | 入口标题、`--model DReSTG`、`--drestg_dataset_id/reference_root/reference_mode/recipe_out`、`--coarse_patch_count`、`--fine_patch_len/stride`、新 setting 标签、`theta_s_init`，全部调用端同步；保留命令行默认的数值与训练循环 | 清理外部身份、提高参数语义可读性，setting 变更只影响输出目录名 |
| `models/TVCFormer.py` → `models/DReSTG.py` | Model 类保持，导入改为新 backbone；参数读取与传递同步改名；`self.model` 路径保留 | 不更改模型主体 API 与计算 |
| `layers/TVCFormer_backbone.py` → `layers/DReSTG_backbone.py` | `DReSTGBackbone`，`fine_patch_len/stride/coarse_patch_count`；`self.backbone` 与 encoder 保留执行顺序；`detail_decoder` → `forecast_decoder`，`detail_patch_total_len` → `fine_flatten_dim`；Reference runtime `cache_kind` 同步改名 | 论文名与 `D*N_F` 的准确语义，forward 的 `permute/unfold/RevIN/flatten` 原样 |
| `layers/TVC_Encoder.py` → `layers/TemporalEncoder.py` | `build_local_heads/insert_local_heads/split_local_heads` 调用转新名；`fine/local/heads` → `fine_tokens/coarse_nodes`；`q_len` → `token_count`；`self.seq_len`（仅 encoder）→ `self.token_count`；`segment_graph_layers` → `st_graph_layers`、`global_refiner` → `global_graph` | 论文表示层命名；保持 `ModuleList` 构造顺序、循环次数、attention、ST reuse 完全相同 |
| `layers/DualGraph.py` → `layers/DualResolutionGraph.py` | `GlobalR2Refiner` → `GlobalGraphModule`；`make_node` → `pool_global_nodes`；`metric_proj` → `relation_projection`；`mG/mL/union` 具名为 global/ST/union candidate mask；`idsG/wg/full/zero/delta/added` 具名；`local_ids/local_r` 具名；`global_mh1` 保留 | 去掉实验 arm 编号；严格保留 `.mean(dim=2)`、`argsort(stable=True)`、top prefix、`torch.nonzero` packed 顺序、Global MH1 差、`lambda`、ST positivity |
| `layers/GraphGenerator.py` → `layers/SpatiotemporalRelation.py` | `graph_generator` → `SpatiotemporalRelationEstimator`；`theta_stat/alpha_stat` → `theta_s/alpha_s`；`pair_r/s_stat/r_init` → `pair_rho/stat_similarity/st_relation_scores`；方法 `_persistent_fusion` → `_fuse_statistical_relations`，`generate_packed_subgraph_features` → `compute_statistical_similarity` | 与公式 `rho,S,alpha_S,Sim^ST` 对齐；保留 clamp/atanh/tanh/ReLU、张量扩展与 cosine dim |
| `layers/Graph_layers.py` → `layers/GraphPropagation.py` | `TVCMixHopPropagation` → `OneHopMixHopPropagation`；`time_node_num` → `coarse_patch_count`（仅该 One-Hop 类）；旧 SAGE helper 代码继续保留但不可触达主模型 | 改公开模型身份；保持 `hop_projection` 注册顺序、双向聚合、index_add、度归一化 |
| `layers/Decoders.py` → `layers/ForecastDecoder.py` | 去旧 TVC 说明；`Flatten_Head` 函数与内部操作保留 | 保持细粒度 token 的 channels-last permute + `Flatten(-2)` + shared Linear |
| `utils/segment_utils.py` → `utils/token_utils.py` | `pool_coarse_patch_nodes`、`interleave_coarse_fine_tokens`、`split_coarse_fine_tokens`、`coarse_node_positions/fine_token_positions`；无运算修改 | 旧 local-head 语义迁移，保留跨界 token 两侧共享池化以及插入位置 |
| `utils/patch_utils.py` | `check_patch_divide` → `compute_patch_layout`；`cut_detail_patches` → `extract_fine_patches`；形参改 `coarse_patch_count/fine_patch_len/fine_patch_stride`，公式原样 | 消除旧命名，保留 ceil 预计算与全历史 unfold floor 计数差异 |
| `data_provider/ref_dataset.py`, `data_provider/data_factory.py` | `TVCReferenceDataset` → `DReSTGReferenceDataset`；`cache_kind` 生产/消费同步 `DRESTG-DUAL-RESOLUTION-V1`；Reference CLI 与 coarse/fine 字段同步 | 保持 `rho_full/statistics/offsets/pair_index/pair_r/packed` payload 形状及其数值 |
| `utils/drestgx_identity.py` → `utils/drestg_identity.py` | recipe/identity schema、fingerprint、screen/fusion/settlement 的**身份字符串**迁移；对应新 argparse 字段，数学和所有数值均不变 | 产物检测必须同步到新身份，防止继续写旧版本标签 |
| `utils/ref_cache.py` | `BASE_SCHEMA/SCREEN_SCHEMA` → `drestg_base_rawrho_v1/drestg_st_rawpos_softk_v1`，`BASE_STATISTICS` 标签移除 TVC；`base/st`、`CONTRACT_VERSION=1`、tie-break 等不变 | 新匿名 Reference 身份；需用新版本生成 meta/ok；旧版不能自动 reuse |
| `scripts/ref.py` | 与 YAML/Reference 新术语、identity 同步；**保留 builder/verifier/npy 精确计算算法、检查强度与发布顺序** | Reference 的机制与数值不改 |
| `exp/exp_main.py`, `exp/exp_basic.py`, `data_provider/data_loader.py` | 模型 import/registry/identity/产物 schema 标签与日志同步；Optimizer、loss、scheduler、样本分割和 DataLoader 选项不动 | 保持结果等价、测试契约自洽 |
| `configs/*.yaml`（7 份） | `model: DReSTG`，命名字段 `coarse_patch_count/fine_patch_len/fine_patch_stride`；其他值逐项保持不动 | 同步论文命名；严禁超参 drift |
| `scripts/DReSTGy_run.sh` → `scripts/DReSTG_run.sh` | 复用旧 runner 的五种模式 `dry-run/prepare/smoke/chain2/full`；参数/路径/产物模式/校验函数全链同步；增加正式运行预检查和明确两槽 ETT 排程；完整成功 cell 删除 checkpoint，完整失败保留证据；设置一致的状态码和完成契约 | 执行层只允许实验可复现性修复，无模型参数变更 |
| `scripts/DReSTGy_cleanup.sh` → `scripts/DReSTG_cleanup.sh` | 严格校验 setting、compact 路径、checkpoint 限定目录；complete：清理 `results/<setting>`、`test_results/<setting>` 与已消费 checkpoint；failed：保留日志/已有 checkpoint 和诊断（不误删追责材料）；幂等；磁盘前后统计 | 适配 experiment-principle-skill 的逐 cell 清理与审计要求 |
| `drestgy_plan.tsv` → `drestg_plan.tsv` | 28-cell ledger 的 model_id 和 reference mode 标签由脚本生成；数值、矩阵不变 | 消除旧命名；避免不一致的过期 plan |
| `datasets/Ett_Series/*.csv`,`datasets/Customs/*.csv` | 原 ZIP 七个 CSV 文件实际是绝对符号链接，全部指向主机 `/root/autodl-tmp/CTM2/dataset/`；保留原相对位置与文件名，替换成长度为 0 的真实占位文件。真实 CSV 必须由使用者在对应位置放入；不得把原主机绝对路径发布到匿名仓库 | 删除外部实例依赖和内部旧项目路径；不杜撰数据，不修改 `data_loader.py` 数据协议；真实训练必须先安装公共 benchmark CSV |
| `README.md` | 重写匿名仓库用途、参数不变、Reference build/reuse、运行命令、并行限制、真实数据需自备、版权与已验证边界 | 去除内部 fork 链路和旧实验分支 |
| `statementDReSTGy.md` | 不进入公开仓库，替换为 `statementDm.md`（单独和 zip 内各一份） | 原 statement 详细记录内部实验版本与移植路径，属于公开身份泄露源 |
| `requirements.txt`,`LICENSE`, PatchTST/RevIN 依赖层 | 保留原文件与数值/算法；必要时只修正明确的旧身份标签 | 防止增删依赖或改变数据处理 |

## 四、Reference 严格消费合同及路径

- `reference/base/<dataset>/<split>` 与 `reference/st/<dataset>/<split>` 原样；`stats [N,C,N_C,2]`，全 Pearson `rho [N,C(C-1)/2]`；`st` 的 `off[N+1] / pair[M,2] / r[M]`，`pair_index[M,3]` 在 collate 时增加 batch index；`rho_full[B,U]` 原样；Reference 行号/序列对齐原样。
- `reference_mode`：**默认 reuse**，严格 verify；第一次运行新 schema 时需显式执行 `DRESTG_REFERENCE_MODE=build bash scripts/DReSTG_run.sh 0 prepare`。构建不读外部 TVC05 路径。`REF_DEVICE=cpu`、`REF_BATCH=64` 原样；未增设 Traffic 特殊磁盘门槛。
- 新 runtime `cache_kind='DRESTG-DUAL-RESOLUTION-V1'` 是内存生产/消费标记；不写入 `rho.npy/stat.npy`。Reference `meta.json/ok.json` 的新 `drestg_*` schema 改名影响复用旧元数据，需要新 build；`BASE_STATISTICS` 文字标签更新但均值/标准差/LayerNorm 完全相同。
- `DRESTG_REFERENCE_ROOT=reference` 默认为相对仓库目录；显式路径须由项目运行用户配置。不得自动指向 TVC05，也不得在缺失资产时退回其他项目。

## 五、两份 Shell 的操作合同

1. **唯一入口、复用稳定链**：只交付 `scripts/DReSTG_run.sh` 和 `scripts/DReSTG_cleanup.sh` 两个实验 Shell，`scripts/ref.py` 继续承担已有 Reference build+verify；不增加独立 launcher、状态汇总器、screen 创建/销毁逻辑。`run.py`、`exp/exp_main.py` 保持既有 train → validation → best-checkpoint reload → test → metrics 路径。
2. **五模式**：`dry-run` 只做静态和 recipe 配置测试，产出新 28-cell ledger；`prepare` 对全部七数据集三分割 build/verify；`smoke` 用 Traffic-96、1 epoch 做单 cell 执行链确认；`chain2` 用 ETTh1-96/192 两并发 1 epoch；`full` 28 正式 cell，完成 28 行 summary。Smoke 允许 1 epoch 是原有验证模式，正式 100 epochs 不变。
3. **严格并发**：ETT 4 数据集 × 4 horizon 共 16 cells，按固定顺序 `(96,192)`、`(336,720)` 配对，任何时刻最多 2 个子任务；每对 `wait` 两个 PID 完成、检查双方退出码、清理/完成合同后进入下一对。剩余 Weather、Electricity、Traffic 12 cells 在所有 ETT pair 完毕后逐个串行，三者均不能与彼此或 ETT 同时运行。第二个 GPU 仅通过可选 `ETT_SECOND_GPU` 指定；默认复用位置参数 GPU（与原 DReSTGy 同一 GPU 双进程行为保持一致），多 GPU 有资源时可显式分流。脚本不得自行判断某数据集以外的并发例外。
4. **完成原子单元**：`dataset + horizon + seed2021`。`cell_identity`、`recipe`、`metrics.json`、`runtime.json`、`formal.log` 和 checkpoint 在**清理前**交叉校验。成功后删除冗余结果/test/checkpoint，`complete.json` 原子发布，记录 `checkpoint_removed=true`；完整 cell 可恢复跳过，已完成产物不再依赖被删除的 checkpoint。**训练失败与清理失败**停止正式矩阵并保留日志/失败现场；不将 summary 计算放入每个 epoch/case 的致命训练链。
5. **确定性查找**：结果目录严格来自 `run.py/build_setting()`，形如 `results/<model_id>_dm...`，而非猜测 `long_term_forecast_` 前缀。checkpoints 为 `<checkpoint_root>/<setting>/checkpoint.pth`。确保唯一结果目录匹配并验证清理目标在仓库根下、安全白名单标识、非软链跳转；失败/中断时不得无条件清理未确认的 checkpoint。
6. **资源**：`MIN_FREE_GB=8`、每 cell 开始前和完成清理后检查剩余空间；磁盘阈值适用于七数据集，不另给 Traffic 高门槛；Reference build 前后同样检查。保留 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` 原脚本合法正整数的默认处理。输出 `models/` 等目录不增加版本并存文件。
7. **手动 screen**：外层用户自行执行 `screen -S drestg_full`，内部执行 `bash scripts/DReSTG_run.sh 0 full 2>&1 | tee drestg_full.log`。脚本不创建 screen。对两个并发 ETT cell 分离 compact 日志，外层主日志只记录状态/进度。
8. **成功、失败和幂等**：`cleanup complete` 只在 metrics 与 runtime 校验后执行，精确删除本 cell 的临时目录及 checkpoint，保留 `metrics.json/runtime.json/formal.log/complete.json`；再次清理同 cell 不触及其他 case；`cleanup failed` 不删 checkpoint 或日志并打印状态；preflight 和正常运行分别可验证。关闭干扰运行的奇异不安全路径。

## 六、验收 Gate（实施前承诺与实施后记录）

- A. 原始 `DReSTGy.zip` 解包作为不可修改基线；新仓库根无中间 `main/`、`DReSTGy/` 层。新旧文件列表逐项映射，排除老 statement，保留许可证与 placeholder 数据路径。
- B. 七 YAML **所有旧超参数值**均经人工/脚本字段名映射后一一一致；28 cell、seed、horizons、per-dataset mass ratio 一致；Reference full path / `st` 分支/算法函数主体完全一致。
- C. `python -m compileall`、`bash -n`、Python imports 与 `dry-run` 全通过；检查旧 TVC 实验身份是否仍遗留于运行代码或公开 README。许可与来源 attribution 除外。
- D. 在 CPU + PyTorch 可用环境，尝试基线对新模型的同 seed、同合成 Reference payload 前向输出、参数数量、训练反传/关键 gradients 逐元素比较。若依赖（如 DGL）缺失，则用**仅存在于本地测试环境的 stub**满足未消费的 SAGE 导入，明确限制，不将 stub 放进仓库。
- E. 做最小 Reference 数值/selector、schema mismatch、packed tensor shape 的离线校验；真实数据构建/真实 GPU 28-case 验收只允许记为“待用户目标环境验证”，不能凭 dry-run 宣称完成。
- F. 使用模拟命令/临时目录独立测试 shell 安全边界、双 ETT 并行对 barrier、三个大数据集严格串行及一例失败退出。任何模拟只代表脚本逻辑验收，不能代表真实训练成功。

> 注意：本 ZIP 所带七个 CSV 实际均为链接至旧计算机 `/root/autodl-tmp/CTM2/dataset/` 的绝对符号链接，压缩包中没有实际 ETTh/ETTm/Weather/Electricity/Traffic 数据；公开版按同名相对路径改为零字节占位文件；本地不能完成真实 GPU 训练和全量 Reference build。所有离线验收将标注测试级别。测试不在仓库中生成真实指标或虚构涨点。

## 七、实施后回填位置

**必须在完成源码修改后**追加最终验收摘要：文件差异、编译和 dry-run 状态、前向等价性、Reference 合同、Shell 测试、已知未验收项与压缩包内部根目录；对任何未通过的项目显式标识并修复后复测。

## 八、实施后审计回填（2026-11-18）

### 8.1 文件、配置与命名 Gate

- 修改基线为原始上传的 `DReSTGy.zip`，在独立只读用途的解包副本上进行旧版对照；其 Python、Shell 和配置值没有被覆盖。最终输出一级根目录严格是 `DReSTG-main/`，下一级直接是 `scripts/`、`configs/` 等，无 `DReSTGy/` 或额外 `main/` 层。
- 七套 YAML 做严格字典比较：仅允许 `model=TVCFormer→DReSTG`、`time_node_num→coarse_patch_count`、`patch_len→fine_patch_len`、`stride→fine_patch_stride` 的键名/标识迁移；所有其他字段、每一个数值以及默认来源保持一致。四份 ETT YAML 各 34 个字段，Weather/Electricity/Traffic 各 31 个字段，**全部通过**。
- Python 全项目 `compileall`、两份 Shell `bash -n`、runner `dry-run` 均通过；`dry-run` 固定产生 28 条唯一 cell 的 seed2021 ledger。公开代码和文档对旧实验标识 `TVC / DReSTGx / DReSTGy / R2-M-U / R2 / O1 / CTM2 / autodl-tmp` 的静态检索为零命中；许可证和合法上游著作权声明保留。
- 原版七个 CSV 路径是绑定原计算机路径的**绝对符号链接**，ZIP 内无数据实体。现已在相同相对路径建立零字节常规占位文件，避免路径泄漏或以错误的符号链接指向其他计算机。真实 GPU 复现之前必须替换成完整公开 benchmark CSV。

### 8.2 Python 数值等价性 Gate

- 使用固定输入、随机种子、相同 dtype/CPU、同一合成 Reference payload，在模型原版和重构版分别运行。临时测试进程针对**未消费的 SAGE 导入**注入最小 DGL stub；该 stub 不进入公开项目包，也不能代替真实 DGL 环境下的训练验证。
- ETTh1（Fine `12/6`、2 层、`D=32`）与 Weather（Fine `24/12`、2 层、`D=64`）均得到**模型初始化参数按注册顺序逐元素相同、eval 前向完全相同、train 前向完全相同、反向梯度完全相同，全部最大绝对误差为 0**。参数总量分别为 203364、295524（测试的对应模型实例）。
- Traffic（Fine `16/8`、3 层、`D=128`）也通过同样的严格逐元素等价性测试。为了控制离线内存，该项仅在测试脚本中把原始通道数 862 缩为合成的 7 通道；未改动任何打包配置，也没有据此声称 862 通道真实 Traffic 训练得到验证。该测试实例原版/新版均为 1045349 参数，85 个 state dict 条目，forward/train/gradient 最大误差为 0。
- 参数属性和 module 命名变化使 `state_dict` **键名有变**；旧 checkpoint 不能直接按 `strict=True` 无映射加载。测试同时核对了按模块注册顺序的初始参数数组，保证变量名迁移未改变新训练的初始化计算。

### 8.3 Reference 数值与契约 Gate

- 使用相同固定生成的九个长度 336、七通道合成样本（包含强正/负相关对和常数通道），分别运行原版和新版 `build_base()`、`build_screen()`、`verify()`。
- 六份 Reference 数组逐元素 **完全相同**：`base/rho.npy` `[9,21]`、`base/stat.npy` `[9,7,7,2]`、`base/idx.npy` `[9]`、`st/off.npy` `[10]`、`st/pair.npy` `[57,2]`、`st/r.npy` `[57]`，均无数值偏差；同一批样本选择了 57 个正相关候选 pair；两版均完成 exact verification。
- `meta.json` 除实施前声明的 schema、source label、耗时等身份字段外的合同字段逐项一致；新 `ref_cache.load_base()` 严格拒绝旧 `drestgx_*` schema，防止误认旧资产为已验证新资产。**正式用户仍必须在真实数据环境执行一次新的 Reference build/verify**；本次没有对真实 benchmark 资产做重新发布。

### 8.4 Shell 生命周期 Gate

- 两份正式 Shell 均通过 `bash -n`。测试在完全独立的临时仓库副本中执行，覆盖非法 checkpoint root 拒绝、精确结果与 checkpoint 删除、其他 cell 产物保留、重复清理幂等、目录穿越拒绝、非授权 compact 路径拒绝、失败时日志留存。
- 使用原 Shell 调度逻辑和临时注入的**无训练 mock** 跑完全量 `full` 控制流：准确启动 28 个不同 cell；峰值并发数恰为 2，且并发窗口内只有 ETT；Weather、Electricity、Traffic 三者各四个 horizon 全部逐个排他执行；正式每个 cell 的参数仍由原 runner 创建。mock 未伪造训练指标，故**仅证明调度与 Shell 控制流**。
- 成功 cell 在删除 checkpoint 前先核验 metrics/runtime/recipe/身份/路径，再原子生成 `complete.json`；再次读取完成状态不依赖已删除的 checkpoint。验证失败时停止并保留日志和已有现场，杜绝错误地把中途中断算作完成。

### 8.5 未完成、边界与发布约束

1. **未完成真实 GPU 训练**：压缩包不含真实七数据集 CSV，当前机器也未运行 AutoDL/PyTorch+DGL 的真实训练周期。因此正式 `smoke`、`chain2`、七数据集三分割 Reference build、28-cell full 的端到端生产验收需在用户具备真实 benchmark 和 CUDA/DGL 的环境执行。不得把离线严格等价性外推为真实论文指标已重跑。
2. **旧 checkpoint/旧 Reference 元数据不直接兼容**：身份 schema 和 torch Module 名均已变。采用项目自己的真实数据重新生成 Reference，在新的仓库从头训练；确需复用旧二进制资产时另做受控迁移审计。
3. **ZIP 只放匿名复现必需源文件与 statement**：不包含原始代码副本、临时测试脚本、模拟 Reference、`__pycache__`、`.pyc`、真实数据、checkpoint 和假结果；内部根目录路径与 `statementDm.md` 目录树合同一致。

总体验收状态：**源代码/符号对齐 PASS；七配置全字段冻结 PASS；模型合成数据数值等价 PASS；Reference 合成数组精确等价 PASS；Shell 静态和模拟调度 PASS；真实数据/GPU 完整周期 NOT RUN（明确保留）。**

- **发布解压可移植性补修**：ZIP 保留 `scripts/*.sh` 的 Unix 执行权限，但 Python `zipfile.extractall()` 或跨平台 GUI 解压可能丢失执行位；既有 runner 一律通过 `bash scripts/DReSTG_cleanup.sh` 调用 cleanup，故 `static_preflight` 改为校验该脚本**存在且可读**，避免仅因解压工具权限行为而失败。无任何训练、Reference 或超参数变动。已在最终解压目录重新通过 `dry-run`。
