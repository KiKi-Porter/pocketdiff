# PocketDiff（重建中）

这里是按主技术规格从零重建的 PocketDiff 包。独立模型不使用旧失败原型；
桥接 adapter 只读加载 `targetdiff-main/targetdiff-main/` 官方代码和权重。
当前已完成独立 apo→holo MVP、版本化 cache、官方 TargetDiff adapter、
完整耦合调度及 reference ligand forward-noise 条件模块：

- `pocketdiff.data.schema.PocketComplex`
- `pocketdiff.data.schema.ResidueMetadata`
- `pocketdiff.data.schema.PocketBatchState`
- `pocketdiff.data.schema.PocketDiffPrediction`
- `pocketdiff.data.schema.PocketStepOutput`
- `pocketdiff.data.apo2mol_adapter.Apo2MolAdapter`
- `pocketdiff.models.PocketDiffModel`
- `pocketdiff.targetdiff.TargetDiffAdapter`
- `pocketdiff.targetdiff.forward_noise_reference`
- `pocketdiff.sampling.pocket_step`

`Apo2MolAdapter` 直接读取 pocket PDB 和 ligand SDF，输出合法的
`PocketComplex`。它用原始文件顺序建立 apo/holo 配对，并逐项检查残基名、
原子名和元素；不会取原子交集、按坐标排序或截断。Apo2Mol 发布文件中
`segment/chain/resseq` 标签存在系统性差异，因此成功样本以 holo 标签生成
规范元数据，并在 `AdapterDiagnostics` 中记录原始标签差异；残基名或原子
签名不一致的样本会被明确过滤。holo pocket 和 ligand 先通过 proper
Kabsch 对齐到 apo 坐标系，再统一减去 apo 蛋白均值。

已运行真实 1000 步耦合 smoke；官方等价性仅验证了短窗口，不能视作完整
1000 步逐步等价证明。Phase 17 已完成固定 8/8 样本、400 步多 k clean
训练；尚未训练 noised 或 self-state 条件，工程流程能运行不代表模型已经学会
可靠的迭代 apo→holo。此次固定时间表的 train/holdout loss 下降，但 k=18/19 回归。
当前进度以 `.codex-tasks/pocketdiff-development/PROGRESS.md` 为准。

当前独立 diffusion core 的正式训练入口为：

```bash
PYTHONPATH=. conda run -n targetdiff python -m pocketdiff.scripts.train_diffusion \
  --output-dir /path/to/run \
  --train-count 24 \
  --holdout-count 16 \
  --updates 480
```

该入口读取 Apo2Mol split，训练 `DiffusionMotionAdapter`，保存
`checkpoint.pt` 和 `report.json`，并执行 strict reload 与 apo-start 连续
reverse sampler 验证。默认仍是 `remaining` 参数化、endpoint loss 权重 1.0、
当前 sampler 时间条件和 ligand→protein vector cross message。

从 checkpoint 做独立 PocketDiff 采样：

```bash
PYTHONPATH=. conda run -n targetdiff python -m pocketdiff.scripts.sample_diffusion \
  --checkpoint /path/to/run/checkpoint.pt \
  --output /path/to/sample_report.json \
  --trajectory /path/to/sample_trajectory.pt \
  --split valid \
  --index 0 \
  --steps 8
```

Phase 64 已验证真实 2/1 小样本 smoke、checkpoint reload 和 2 步连续采样；
这只是工程入口验证，不代表模型已经科学上学会 apo→holo。

`forward_noise_reference(adapter, reference, t_graph, generator=...)` 使用固定
apo 中心化后的 clean reference state，返回只替换 ligand 坐标和类别的新 state。
每次从 clean reference 独立加噪，支持每图不同时间；不要对前一次加噪结果再次调用。
该模块复用已加载的官方 schedule，不生成新监督标签。

`pocketdiff.training.build_bridge_batch(clean_batch, pocket_k=...)` 已支持每图
`k=0…19` 的 teacher-forced 输入、当前局部 remaining 标签和 `P_(k+1)`
坐标监督；也可提供独立 `generator` 均匀采样 k。`remaining_steps` 是逐图
张量，调用几何 solver 时须用 `batch_residue` 映射到逐残基。
`MultiKCleanTrainer` 每步从端点重新采样 k 并构造上述输入；
`evaluate_multik_clean` 固定枚举 20 个 k，报告 teacher-forced 观测。

SO(3) 小角度精度修正后，缓存几何版本为 `geometry-v2`。旧 `geometry-v1`
样本和 manifest 会明确拒绝加载，需要从源结构重新生成目标标签后用于新训练。
历史 checkpoint 和结果保留；Phase 17 新权重保存于
`.codex-tasks/pocketdiff-development/phase17-multik-clean-training/raw/run/checkpoint_0400.pt`。
该模型从头训练；loss、k 抽样记录、各时间评估和曲线在同一目录。
Phase 17 模型的晚期预测曾回归，后续Phase 18修正见下文；不能把 teacher-forced 评估当作 20 步自主推理效果。

Phase 18 增加显式 `motion_parameterization="bridge_rate"`：令
`r=(20-k)/20`，head 输出对应 `remaining/r`，公开预测乘回 r，仍为实际
remaining 平移/旋转，solver 接口不变。`MultiKCleanTrainer` 为该模式使用
配对的 `masked_bridge_rate_loss`；优化日志 `loss` 为归一化单位，
`remaining_loss` 和固定20k评估为实际 remaining 单位。旧配置默认
`remaining`，原权重保持可加载；新 checkpoint 显式记录参数化模式。
旋转上界相应变为 `πr`，当前仅验证 teacher-forced clean bridge，自主状态
偏离理想桥时的纠偏能力需单独验证。仅直接加载 state_dict 无法识别模式，
必须一并使用 checkpoint 内的 model_config。

复现 Phase 18 的单次对照（复用 Phase 17 的只读缓存）：

```bash
conda run -n targetdiff python -m pocketdiff.scripts.phase18_rate_experiment \
  --output-dir /path/to/empty-phase18-run
```

固定400步对照中，Phase 18 留出集k19一步bridge误差从原模型0.061072降到
0.010778 Å（零更新为0.020155 Å）；k0 holo RMSD仅0.670852→0.669965 Å。
Phase18时126项测试通过，checkpoint重载输出误差0；后续自主20步结果见Phase19记录。
新权重保存在 `.codex-tasks/pocketdiff-development/phase18-late-step-diagnosis/raw/run/checkpoint_0400.pt`。

诊断、阶段结果和下一步见
`.codex-tasks/pocketdiff-development/phase18-late-step-diagnosis/`。
每100步保存含优化器/RNG的 checkpoint，底层
`MultiKCleanTrainer.from_checkpoint(path, clean_batch)` 支持精确续训；该固定对照
脚本本身只接受空输出目录，不覆盖历史运行。

运行或恢复 Phase 17：

```bash
# 新输出目录只重建固定的 16 个 geometry-v2 样本，避免覆盖历史实验。
conda run -n targetdiff python -m pocketdiff.scripts.phase17_multik_train --output-dir /path/to/new-run

# 同一数据/目录从已保存步继续；总步数用 --steps 指定，模型和优化器配置沿用 checkpoint。
conda run -n targetdiff python -m pocketdiff.scripts.phase17_multik_train \
  --output-dir /path/to/new-run --resume /path/to/new-run/checkpoint_0100.pt --steps 400
```

验证：

```bash
conda run -n targetdiff python -m pytest -q pocketdiff/tests

# Phase 1 adapter tests and a 50-sample-per-split raw smoke
conda run -n targetdiff python -m pytest -q pocketdiff/tests
PYTHONPATH="$PWD" conda run -n targetdiff python pocketdiff/scripts/apo2mol_smoke.py \
  --limit 50 \
  --output .codex-tasks/pocketdiff-development/phase1-data/raw/adapter_smoke_report.json
```


Phase 19 已完成固定clean配体的自主20步评估（未训练）：

```bash
conda run -n targetdiff python -m pocketdiff.scripts.phase19_clean_rollout \
  --output-dir /path/to/empty-phase19-run
```

`pocketdiff.sampling.run_clean_rollout(model, inputs)`只接受
`pocketdiff.sampling.clean_rollout.INFERENCE_FIELDS`定义的apo/特征/映射/配体输入；
内部计算apo有效frame并按k0..19更新，仅将预测坐标传入下一步。返回21个坐标状态、
frame有效性与20次运动。holo与标签不进入该接口。
`pocketdiff.evaluation.clean_rollout.score_clean_rollout`在推理结束后使用固定apo有效
原子集合评分，不重新对齐，不按预测mask筛掉难例。

真实固定8/8对照的终点全原子holo RMSD：训练组apo/Phase17/Phase18为
0.585246/0.604356/0.587652 Å，留出组为0.671782/0.685551/0.676442 Å。
Phase18减轻了回归，但平均全原子效果仍劣于不更新apo；留出组5/8改善、3/8回归。
136项测试通过、48条20步轨迹完整、输入/旧权重不变，属于工程通过，尚未达成学习目标。
完整轨迹/分步指标/曲线位于
`.codex-tasks/pocketdiff-development/phase19-clean-rollout/raw/run/`；分析和下一步见该阶段`RESULTS.md`。
下一阶段先建立detached self-state监督输入契约，量化标签尺度与bridge_rate旋转范围，
再决定训练对照；不使用holo挑选提前停止状态。


Phase 20 新增 `training.build_self_state_batch`：给定 Phase19 自主轨迹的
`positions[0..20]` 和每图 `pocket_k`，生成 detached 当前状态并重新计算
current→holo 标签。轨迹 step0必须等于apo；model_kwargs不包含holo或标签。
真实审计显示k19自主标签常超出bridge_rate的πr旋转范围，归一化目标也在晚期放大，
因此bridge_rate目前只作为teacher-forced clean模式保留，不能直接用于self-state训练。


Phase 21 确定：自主 self-state 不使用 `bridge_rate` 的 `r` 归一化和 `πr`
旋转界，而是直接使用当前自主结构到 holo 的物理 remaining SE(3) 标签。
`SelfStateBatch` 和 `masked_self_state_motion_loss` 已通过真实 Phase18轨迹
的混合k forward/loss/gradient审计。下一阶段进行有限self-state训练，直接评价
自主apo→holo，而不是继续扩大桥接工程。

## Phase31：独立current χ和刚体＋χ oracle

推理几何入口为`pocketdiff.geometry.build_current_chi_state`，只接收当前坐标和原子/残基拓扑，输出当前χ与geometry_rotatable_mask。`oracle_rigid_chi_reconstruction`为holo监督诊断入口，单独提供named-atom supervision_mask与排除歧义slot的training_safe_mask；不能将oracle标签传到推理模型。

16真实样本平均apo/rigid/rigid+χ holo RMSD：1.027622/0.855461/0.221056 Å（oracle，非学习结果）。当前新状态尚未接入旧模型/训练器；旧head依然是chi_apo语义。详细结果与恢复边界见`.codex-tasks/pocketdiff-development/phase31-oracle-current-chi/RESULTS.md`。桥接与微调暂缓，继续完善独立本体。
