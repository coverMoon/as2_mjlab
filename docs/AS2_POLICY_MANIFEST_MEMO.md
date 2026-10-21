# AS2 Policy Manifest — 设计备忘

**状态：** 导出端 v1 已实施；C++ 部署端与逐项跨语言一致性测试待实施。

**用途：** 为 `as2_mjlab` 导出的策略与 `as2_control` 的部署端建立可验证的输入、输出和时序契约。当前格式、命令和边界见 [Policy Manifest v1](POLICY_MANIFEST_V1.md)。本文保留总体设计方向；按当前要求，不加入模型哈希。

## 1. 为什么需要它

随着 PPO、HIM/RMA、Rough、视觉等策略迭代，观测项、顺序、维度、时间窗口、动作语义可能不同。只保存 `policy.onnx` 和手工编写的 YAML 容易出现“张量维度相同但含义不同”的部署错误。

目标是让每个导出策略包含可检查的描述文件：

```text
policies/as2_flat_v1/
├── policy.onnx
└── manifest.yaml
```

**来源原则：** `as2_mjlab` 训练/导出代码生成 Manifest，是事实来源；`as2_control` 启动时读取、验证、构建观测执行表。部署侧不得根据机器型号自行猜测关节或观测顺序。未来支持多模型输入时，可扩展为按张量分别描述的格式。

## 2. 拟记录的最小字段

| 分组 | 建议字段 | 目的 |
|---|---|---|
| 身份与版本 | `schema_version`、`robot`、训练 Git Commit、MjLab/RSL-RL 版本 | 明确来源和契约版本 |
| 模型 | ONNX 路径、输入输出名称、dtype、shape | 启动时验证张量与模型文件一致 |
| 观测 | 有序 Term 列表、各项 `name`、`dim`、参数、坐标系、单位、数据来源 | 明确拼接和计算语义 |
| 预处理 | 每项 clip、scale、历史顺序、是否包含归一化于 ONNX | 避免错位与重复归一化 |
| 时间状态 | 策略 `dt`、phase 公式和重置规则、上一动作的定义与更新时序 | 保证时序一致 |
| 关节与动作 | **策略动作顺序**的关节名、默认关节位置、action scale/offset、输出控制类型 | 按名称映射到 SDK2 电机索引 |
| 数据依赖 | 必需的本体状态、外部感知类型及失效语义 | 禁止缺失观测时静默填零 |

**不建议放入 Manifest：** DDS 域/网卡、硬件急停、通信超时、实机校准、机械安全上限等部署安全配置。这些应属于 `as2_control` 的设备与安全配置，不能让训练模型覆盖。

## 3. AS2 Flat 基线（以当前仓库代码为参照）

任务统一使用 `as2-<terrain>-<algorithm>`：

| 任务 | Actor / Critic | 接口特点 |
|---|---|---|
| `as2-flat-unitree` / `as2-rough-unitree` | 47 / 261 | 保留现有 gyro 观测、phase 和 Unitree 运动先验 |
| `as2-flat-ppo` / `as2-rough-ppo` | 45 / 259 | 根部机身角速度，无 phase，无 Actor 地形输入 |

`as2-flat` / `as2-rough` 不再注册。旧 checkpoint 使用对应 `unitree` 任务加载，不能作为新 PPO 的 45 维 checkpoint 直接加载。

以下 47 维顺序属于 Unitree baseline。

Actor 观测顺序：

```text
base_ang_vel(3) → projected_gravity(3) → command(3)
→ phase(2) → joint_pos(12) → joint_vel(12) → actions(12)
总计 47 维，策略输出 12 维 raw action；策略周期 0.02 s。
```

需要精确描述的语义：

- `base_ang_vel`：当前训练读取 MuJoCo 内置 IMU gyro，不可不经坐标检查就替换成 SDK2 原始角速度。
- `projected_gravity`：基于根部机身姿态的机体系重力投影，不是线加速度计读数。
- `command`：策略实际使用的三维机体系速度指令；`phase` 的站立判断使用三维命令的 L2 范数 `< 0.1`。
- `phase`：`phi = ((episode_step * 0.02) mod 0.6) / 0.6`，非站立时输出 `[sin(2πphi), cos(2πphi)]`，站立时输出 `[0,0]`；Reset 观测使用 `episode_step=0`。
- `joint_pos`：未加入 encoder bias 的关节角减各关节默认位置；`joint_vel`：关节速度减默认速度（当前默认值为零）。
- `actions`：在**下一次策略推理的观测**中使用前一次策略产生的**原始动作**，不使用 PD 目标关节角或 0.25 缩放后的动作；初次 Reset 为零。
- `q_target = q_default + 0.25 * raw_action`；**关节顺序必须依据训练时的实际 action/joint 映射输出，不应从足端 Site 顺序推断**。
- 训练 Actor 启用 `obs_normalization=True`。目前所检查的 RSL-RL MLP ONNX 导出实现内含归一化模块；后续应以**实际生成的 ONNX 图及数值对比**确认，部署端不得重复归一化。

当前 Flat 与 Rough 的 Actor 均为上述 47 维，`height_scan(187)` 仅输入 Critic。未来如果给 Actor 追加扫描，需要记录 RayCast 网格、顺序、坐标系、命中/未命中值、缩放及外部感知来源。不能只用 `dim: 187` 代替计算语义。

新 PPO baseline 顺序为 `base_ang_vel(3) → projected_gravity(3) → command(3) → joint_pos(12) → joint_vel(12) → actions(12)`，共 45 维，动作仍为 12 维、50 Hz。角速度来自 `root_link_ang_vel_b`；角速度 scale 为 0.25，命令 scale 为 `[2,2,0.25]`，关节速度 scale 为 0.05。它的 Critic 追加真实线速度 3、扫描 187、足端世界高度 4、接触状态 4、当前连续触地时长 4、接触力 12，共 259 维。具体配置见 [PPO Baseline](PPO_BASELINE.md)。

按本轮后续确认，保留上一轮已实现的 Manifest 功能并兼容 PPO。PPO 使用通用 MjLab Runner，其 checkpoint 不额外存档训练契约，也不自动导出 ONNX；独立 `export` 可生成 45 维模型及 Manifest，并如实标注契约来自导出环境。

## 4. 未来文件格式示意（不是已实现 Schema）

```yaml
schema_version: 1
robot: as2
model:
  file: policy.onnx
  input: {name: <from-onnx>, shape: [1, 47], dtype: float32}
  output: {name: <from-onnx>, shape: [1, 12], dtype: float32}
  normalization: embedded_in_onnx  # 需根据实际导出文件验证
policy_dt: 0.02
observation:
  ordered_terms:
    - {name: base_ang_vel, dim: 3, source: imu_gyro, frame: body}
    - {name: projected_gravity, dim: 3, source: base_orientation, frame: body}
    - {name: command, dim: 3, source: active_velocity_command, frame: body}
    - {name: phase, dim: 2, period: 0.6, command_threshold: 0.1}
    - {name: joint_pos, dim: 12, kind: relative_to_default}
    - {name: joint_vel, dim: 12, kind: relative_to_default}
    - {name: actions, dim: 12, kind: previous_raw_action}
action:
  kind: joint_position_residual
  scale: 0.25
  joint_names: <ordered-names-from-export>
  default_joint_pos: <ordered-values-from-export>
```

说明：示例里的 `source` 和 `kind` 是拟议的稳定语义标识，并非当前 MjLab 原生 Metadata 的字段；占位符必须在正式导出时替换。未来还需明确逐项 clip/scale、reset 与 history 的机器可读规则。

## 5. 实施顺序

1. 在 `as2_mjlab` 导出阶段读取 **实际生效**的 Actor 观测顺序、Action 顺序、预处理以及 ONNX 张量，生成 Manifest（v1 已实施）。
2. `as2_control` 启动时加载 Manifest，并与 ONNX 输入输出、RobotModel、可用 ObservationTerms 逐项验证；不匹配则拒绝启动 RL。
3. 用相同的确定性仿真状态/命令/上一动作，对比 MjLab 与 C++ 的**每项观测、最终输入张量和 ONNX 推理输出**，而不是只对比维度。
4. 先覆盖当前 Flat/Rough 47-D，再扩展未来带高度扫描的 Actor、历史窗口和视觉等多输入模型。

## 6. 仍待验证的契约细节

- SDK2 IMU 坐标系、四元数方向与 MuJoCo 根部/IMU Site 的一致性；必要时的固定变换。
- 训练端实际 Actor/Action 关节顺序到 SDK2 电机索引的映射，以及全部默认关节角数值。
- 推理循环的首帧相位、第一次输出动作及历史重置时序；需要对照训练环境逐步验证。
- ONNX 中是否已包含预处理/归一化，是否包含动作后处理；以实际导出图为准。
- 站立、传感器失效、策略停用与重新启动时，状态型观测项的明确重置规则。

**当前范围：** 已实现训练契约存档、ONNX 旁的 Manifest 自动生成和 Python 校验；C++ 解析、SDK2 映射与插件体系不在本仓库第一版范围内。
