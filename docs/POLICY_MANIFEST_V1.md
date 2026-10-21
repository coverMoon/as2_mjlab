# Policy Manifest v1

`export` 生成 ONNX 的同时生成部署契约。每个模型使用同名侧边文件，保留现有导出命名规则：

```text
model_2000.pt
model_2000.onnx
model_2000.manifest.yaml
```

```bash
uv run export as2-flat-unitree --checkpoint-file logs/rsl_rl/as2_velocity/<run>/model_2000.pt
uv run export as2-rough-unitree --checkpoint-file logs/rsl_rl/as2_velocity/<run>/model_2000.pt --output-file /tmp/rough.onnx
uv run export as2-flat-ppo --checkpoint-file logs/rsl_rl/as2_ppo/<run>/model_2000.pt
uv run python -m as2_mjlab.policy_manifest /tmp/rough.manifest.yaml
```

支持四个任务：`as2-flat-unitree` / `as2-rough-unitree` 的 Actor 为 47 维，`as2-flat-ppo` / `as2-rough-ppo` 为 45 维，输出均为 12 维。Critic 的 187 点高度扫描不属于部署 Actor 的输入。Unitree 新训练正常保存 checkpoint 时，也会在原有自动生成的运行目录名 ONNX 旁写入对应 Manifest；PPO 使用通用 Runner，仅通过独立 `export` 导出。

## 文件内容

| 字段 | 含义与来源 |
|---|---|
| `schema_version` | 当前为整数 `1`；消费者须拒绝不支持的版本 |
| `robot` / `task_id` | AS2 和本次任务标识 |
| `model.file` | 与 Manifest 同目录的 ONNX 文件名 |
| `model.inputs` / `outputs` | 从 ONNX 图读取的名称、dtype、shape 列表；第一版为单个固定批量 `[1, dim]` float32 张量 |
| `policy_dt` / `clock` | 实际控制周期、episode 计数及上次动作的重置时序 |
| `observation.ordered_terms` | 从运行时 ObservationManager 读取的顺序、连续偏移、维度、语义和预处理 |
| `observation.normalization` | 已加载 Actor 的归一化类型、位置及 epsilon；归一化在 ONNX 内，部署端不能重复执行 |
| `action` | 从实际 ActionTerm 读取的策略输出关节顺序、默认角度/速度、scale、offset、clip、名义 PD 增益 |
| `requirements` | 缺失或非有限观测必须拒绝；当前 Actor 无外部地形扫描依赖 |
| `provenance` | checkpoint 文件名、契约来源，以及分别记录的训练/导出 Git commit、工作区是否有改动和软件版本 |
| `verification` | 本次 PyTorch/ONNX 推理对比的样本数、容差和最大绝对误差 |

不记录模型哈希，不保存 DDS 网卡、SDK2 电机编号、实机校准或硬件安全上限。关节应由部署端按名称映射到它的 RobotModel。

## 观测与动作的精确语义

每个观测项记录 `name`、`offset`、`dim`、`kind` 和计算语义。`offset` 指向归一化前的拼接输入，不能根据维度猜测项的意义。

处理顺序为 `compute → clip → scale → concatenate → ONNX normalization`。部署端负责前四步，ONNX 执行已学习的归一化和确定性网络推理。训练的随机观测噪声不在部署端重现；第一版只支持无历史、无延迟的平面拼接，遇到未知观测函数、历史窗口、延迟、非 MLP 或多输入会明确拒绝。

当前七类语义：

- `imu_angular_velocity`：MuJoCo gyro 的三轴输出，单位 rad/s。记录真实 sensor/site/父 body、site 在父 body 内的位置和 wxyz 四元数。SDK2 数据需要经过实际坐标验证，不能直接当作任意机体系角速度。
- `base_angular_velocity`：PPO 使用的根部机身角速度 `root_link_ang_vel_b`，记录其机身坐标系，单位 rad/s；不能与任意安装方向的 IMU 原始数据混用。
- `projected_gravity`：将世界单位向量 `[0, 0, -1]` 逆旋转到根部机身坐标系。
- `velocity_command`：当前实际生效的 `[lin_vel_x, lin_vel_y, ang_vel_z]`，单位分别为 m/s、m/s、rad/s。heading 控制器转换后的角速度已经包含在该命令中。
- `gait_phase_sin_cos`：`phi=((episode_step*policy_dt)%period_s)/period_s`，输出 `[sin(2πphi), cos(2πphi)]`。三维命令 L2 范数严格小于 0.1 时输出零；reset 时 step 为零，非站立首帧为 `[0,1]`。
- `joint_position_relative` / `joint_velocity_relative`：按各自记录的关节顺序计算 `measured-reference`；不加入训练的 encoder bias。
- `previous_action`：上一次推理结果经过 wrapper clip 后、进入 action scale/offset/target clip 前的动作；reset 时全部为零。它对应生成下一次观测时 ActionManager 中最近接收的动作，不能再额外延迟一个周期。

ActionTerm 的顺序可能与机器人自然关节顺序、关节观测顺序不同，分别记录，不能互相代替。部署计算：

```text
applied_action = clip_wrapper(onnx_output)       # wrapper_clip 为 null 则直接使用
q_target = clip_target(offset + scale * applied_action)
previous_action = applied_action               # 供下一次观测使用
```

`target_clip` 在 scale/offset 之后执行，null 表示无该层 clip。仿真的 encoder bias 是训练扰动，模拟器会从目标中减去它；部署端不能照搬随机扰动，也不能以 Manifest 覆盖设备校准。

## 训练契约与旧 checkpoint

Unitree 的 AS2 runner 在初始化时捕获有效部署契约，在每个 checkpoint 的 `infos.policy_manifest` 中保存 `schema_version`、`contract` 和当时的源码/版本信息。导出时将它与当前有效契约逐字段比较；即使输入维度相同，改动观测顺序、phase 周期、默认位置或 action scale 也会拒绝导出并指出字段路径。新 PPO 使用通用 `MjlabOnPolicyRunner`，不额外存档该字段，因此 PPO Manifest 目前标记为 `export_environment`，`--require-training-contract` 校验会拒绝它。

旧 checkpoint 没有这一存档，仍可导出；Manifest 标注：

```yaml
provenance:
  contract_source: export_environment
  training_contract_verified: false
  training: null
```

这表示它描述当前导出环境，无法证明旧训练使用了相同语义；不会把当前 Git commit 当作训练 commit。部署流程如要求训练时契约，使用：

```bash
uv run python -m as2_mjlab.policy_manifest model_2000.manifest.yaml --require-training-contract
```

新 checkpoint 导出时则标注 `contract_source: checkpoint`、`training_contract_verified: true`。这证明存档契约与导出环境匹配；dirty 工作区通过 `git_dirty` 如实记录。

## 校验与数值验证

导出先写临时目录，完成 ONNX checker、Manifest 结构检查和独立 ONNX ReferenceEvaluator 与已加载 PyTorch Actor 的数值对比，再发布 ONNX 和 YAML。对比包含一个真实环境观测、全零输入及两组固定随机输入；覆盖已加载的归一化，容差为 `rtol=1e-4, atol=1e-5`。这些输入均为经过观测预处理、尚未归一化的网络输入。

Python 校验器检查 ONNX 输入输出匹配、连续观测切片、维度、关节映射长度与重复、支持的语义/布局等。部署端可以将自身解析得到的完整期望契约传入 `validate_manifest(..., expected_contract=...)`，逐字段比较，拒绝相同维度但不同语义的模型。

独立命令执行结构和契约检查；`verification` 记录的是导出时对比结果，它不会在校验时重新加载 PyTorch checkpoint，也不认证模型权重。C++ 部署端仍需实现本地模型张量、RobotModel、ObservationTerms 能力及控制时序的验证，以及确定性状态的逐项观测对照。
