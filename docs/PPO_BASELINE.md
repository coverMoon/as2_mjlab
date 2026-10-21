# AS2 Baselines

| 任务 | 配置来源 | Actor / Critic | 实验目录 |
|---|---|---|---|
| `as2-flat-unitree` | 保留 `tasks/velocity` 的 Flat 实现 | 47 / 261 | `logs/rsl_rl/as2_velocity` |
| `as2-rough-unitree` | 保留 `tasks/velocity` 的 Rough 实现 | 47 / 261 | `logs/rsl_rl/as2_velocity` |
| `as2-flat-ppo` | 独立 `tasks/ppo`，Plane | 45 / 259 | `logs/rsl_rl/as2_ppo` |
| `as2-rough-ppo` | 独立 `tasks/ppo`，生成地形 | 45 / 259 | `logs/rsl_rl/as2_ppo` |

没有旧名称别名。旧 AS2 checkpoint 应通过对应 `unitree` 任务加载；不转换成 PPO checkpoint。

## 使用

```bash
uv run train as2-flat-ppo --env.scene.num-envs 4096 --agent.logger tensorboard
uv run train as2-rough-ppo --env.scene.num-envs 4096 --agent.logger tensorboard
uv run train as2-flat-unitree --env.scene.num-envs 4096 --agent.logger tensorboard
uv run train as2-rough-unitree --env.scene.num-envs 4096 --agent.logger tensorboard

uv run play as2-flat-ppo --checkpoint-file <ppo-checkpoint.pt> --viewer viser
uv run play as2-rough-unitree --checkpoint-file <old-unitree-checkpoint.pt> --viewer viser
uv run export as2-flat-ppo --checkpoint-file <ppo-checkpoint.pt>
```

四个任务都支持现有 train / play / export 入口。新 PPO 使用通用 `MjlabOnPolicyRunner`，不随保存自动导出 ONNX；独立 `export` 同目录生成同名 ONNX 和 Manifest。PPO 当前不额外存档训练时契约，Manifest 会标注来自导出环境。按后续用户确认保留 Manifest 功能，本轮只增加 PPO 角速度语义支持；C++ 部署端不在本轮范围内。

## PPO 输入与控制

Actor 顺序固定为 `base_ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), actions(12)`，共 45 维。没有机身线速度、相位、扫描、历史、RNN 或 Encoder。

角速度读取真实机身坐标系 `root_link_ang_vel_b`。位置为相对默认关节位置，速度为相对默认速度（当前为零）；`actions` 是上次原始策略输出，未缩放为 PD 位置目标。PPO 配置不做 wrapper action clipping。

沿用 legged_gym 本体观测的基础 scale：角速度 0.25、命令 `[2,2,0.25]`、关节速度 0.05，其余 1。Actor 的噪声在 scale 前应用；Critic 使用对应无噪声项。Critic 扫描沿用现有 AS2 的 187 点和 0.2 scale。

Critic 按顺序追加：

| 观测 | 实测维度 | 来源 |
|---|---:|---|
| Actor 对应项 | 45 | 无噪声本体观测 |
| base_lin_vel | 3 | 真实机身线速度，scale 2 |
| height_scan | 187 | yaw 对齐的 1.6 × 1.0 m 网格，分辨率 0.1 m |
| foot_height | 4 | 足端 site 的世界 Z；奖励用的局部净空不加入观测 |
| foot_contact | 4 | 是否触地 |
| foot_contact_time | 4 | 当前连续触地时长（秒），区别于 Unitree 原有的空中时长 |
| foot_contact_forces | 12 | 四脚三轴力，沿用 MjLab 的 `sign(f) * log1p(abs(f))` |
| 总计 | **259** | |

共用 AS2 MJCF、初始姿态和 Hip/Thigh/Calf 执行器；`q_target = q_default + 0.25 * action`。仿真步长 0.005 s、decimation 4、策略周期 0.02 s，训练 episode 20 s。Actor/Critic 均为 `[512,256,128]`、ELU、启用观测归一化；PPO 超参数与现有 Unitree 配置相同。

## PPO 奖励

基础十项参考 [ETH legged_gym 的基础奖励实现](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) 和 [默认配置](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)。后续加入独立重写的 pose 和局部地形 Foot Clearance，共十二项：

| 名称 | 权重 | 原始值 |
|---|---:|---|
| tracking_lin_vel | 1.0 | `exp(-sum((cmd_xy-v_body_xy)^2)/0.25)` |
| tracking_ang_vel | 0.5 | `exp(-(cmd_yaw-w_body_z)^2/0.25)` |
| lin_vel_z | -2.0 | `v_body_z^2` |
| ang_vel_xy | -0.05 | `sum(w_body_xy^2)` |
| orientation | -0.2 | `sum(projected_gravity_xy^2)` |
| pose | 1.0 | `exp(-mean(((q-q_default)/std)^2))`，容差按速度指令分档 |
| foot_clearance | -0.25 | `sum(abs(local_height-0.10) * norm(foot_world_velocity_xy))` |
| torques | -1e-5 | MjLab `joint_torques_l2` |
| joint_acc | -2.5e-7 | MjLab `joint_acc_l2` |
| action_rate | -0.01 | 原始策略动作差的平方和 |
| joint_pos_limits | -10.0 | 超出软关节限位的距离之和 |
| termination | -200.0 | 失败且没有同时超时 |

速度奖励自行实现，因为当前 MjLab Velocity 的跟踪函数还包含 Z 或 Roll/Pitch 误差，不符合这个基线的独立跟踪形式。其余正则化优先使用通用函数。没有 gait、air-time、slip、soft-landing、stand-still 或 angular-momentum 奖励。

`pose` 在 PPO 内独立实现，不导入 velocity 的奖励或配置。参数采用 velocity 当前最终配置：权重 1.0，按 `norm(command_xy) + abs(command_yaw)` 在 0.1 / 1.5 处分为站立、行走和跑动；Hip/Thigh/Calf 容差依次为站立 `[0.05,0.1,0.15]` rad，行走和跑动均为 `[0.15,0.35,0.5]` rad。Flat / Rough 共享该项，观测和动作接口不变。

`foot_clearance` 在 PPO 内独立返回正 cost，Flat / Rough 均使用目标净空 0.10 m、权重 -0.25。仅在 `norm(command_xy) + abs(command_yaw) > 0.1` 时启用，世界系足端 XY 速度为零时该脚不贡献代价。函数内不取负号、不乘 dt，也不判断步态或摆动相。

奖励使用共享 `as2_mjlab.sensors.TerrainProjectionSensor`：每脚中心加周围四点（半径 4 cm），取最小净空，即以局部最高地形为参考。传感器从静态地形上方竖直查询，净空按实际足端 Z 减命中地面 Z 计算，允许台阶高于脚时得到负值。该传感器不加入 Actor 或 Critic；Critic 原有的足端高度与 187 点扫描保持原样。Unitree 的旧传感器导入路径保留，传感器实现和任务参数不变。

具体差异与时序：

- RewardManager 统一执行 `raw_value * weight * step_dt`，权重只填一次，函数内不再乘 dt。失败终止在 0.02 s 下贡献 `-200 * 0.02 = -4`；纯超时以及失败/超时同时发生均为零，匹配 legged_gym 的 `reset & ~timeout` 语义。
- AS2 当前为一关节一个单位传动比执行器，因此 MjLab 的 actuator-force 平方和对应关节 PD 力矩平方和。
- `joint_acc_l2` 使用 MuJoCo 当前物理子步的关节加速度；legged_gym 使用相邻策略步关节速度差除以策略 dt。第一版明确保留 MjLab 通用实现，不增加有状态奖励缓存。
- 直接使用带符号的奖励总和；未引入 legged_gym 默认的总奖励正值裁剪。软关节限位保留共享机器人定义的 0.9 系数。

## Flat / Rough 与课程

Flat 为 Plane，不启用地形课程。Rough 从 MjLab 通用地形集合独立复制并显式覆盖为当前 AS2 地形：10 层、8 × 8 m 地块，楼梯单级高度 0–0.30 m，初始难度上限 5；类型比例依次为平地 0.1、上楼梯 0.3、下楼梯 0.2、正/反斜坡各 0.1、随机粗糙 0.1、波浪 0.1。

Rough 按距出生点位移大于半个地块升级，低于期望行走距离一半降级；初次 reset 不升级。Play 保留相同的整齐地形生成方式，关闭难度更新，并在 root reset 前重新选择地形。

两个 PPO 任务共享足端摩擦、encoder bias、base COM 偏移、reset 和随机推力配置。Rough 的非足端接触监测排除小腿碰撞体，允许小腿碰楼梯；Flat 保留小腿接触终止，均继续执行相应碰撞物理。

指令课程独立定义在 PPO 工厂，通过 `env_cfg.py` 顶部的 `ENABLE_COMMAND_CURRICULUM = True` 默认开启，Flat / Rough 训练共用。阶段为 `0/500/1000/1500 × num_steps_per_env`，继承现有 AS2 的阶段范围作为初始值；在 reset 时根据全局策略步计数切换范围。Play 关闭课程、观测噪声和随机推力。先验证基础 PPO 在 Flat 的学习趋势，再训练 Rough；本轮未按训练性能调整任何奖励权重。
