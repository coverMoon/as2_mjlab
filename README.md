# AS2 MjLab

MjLab 1.6.0 上的 AS2 速度跟踪任务。

| 任务 | Baseline | Actor / Critic |
|---|---|---|
| `as2-flat-unitree` | 原有 Unitree Flat | 47 / 261 |
| `as2-rough-unitree` | 原有 Unitree Rough | 47 / 261 |
| `as2-flat-ppo` | 独立基础 PPO Flat | 45 / 259 |
| `as2-rough-ppo` | 独立基础 PPO Rough | 45 / 259 |

```bash
uv run train as2-flat-ppo --env.scene.num-envs 4096 --agent.logger tensorboard
uv run play as2-flat-ppo --checkpoint-file <checkpoint.pt> --viewer viser
uv run export as2-flat-ppo --checkpoint-file <checkpoint.pt>
```

旧的 `as2-flat` / `as2-rough` 名称已移除。旧 checkpoint 使用对应的 `unitree` 任务，不转换为 PPO 策略。

配置、奖励与课程说明见 [PPO Baseline](docs/PPO_BASELINE.md)；导出契约见 [Policy Manifest v1](docs/POLICY_MANIFEST_V1.md)。
