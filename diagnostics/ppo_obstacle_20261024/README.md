# PPO 越障停滞专项诊断

最终结论见 [main_02/summary.md](main_02/summary.md)。生产配置、模型权重、已有训练日志均未修改；原有未提交SLR环境展开改动保留。

## 复现实验

从仓库根目录执行，输出目录必须是新目录，避免覆盖实验：

```bash
uv run python diagnostics/ppo_obstacle_20261024/diagnose.py \
  --checkpoint logs/rsl_rl/as2_ppo/2026-10-22_16-09-58_as2-rough-ppo/model_10000.pt \
  --output diagnostics/ppo_obstacle_20261024/new_run \
  --repeats 8 --seconds 20 --seed 20261024 --device cuda:0
uv run python diagnostics/ppo_obstacle_20261024/analyze.py \
  diagnostics/ppo_obstacle_20261024/new_run
```

另采集相同参数至另一个目录，然后生成精简复跑统计和审计：

```bash
uv run python diagnostics/ppo_obstacle_20261024/analyze.py <replay-directory> --metrics-only
uv run python diagnostics/ppo_obstacle_20261024/audit.py <primary-directory> <replay-directory>
```

CUDA路径不承诺长轨迹逐位一致。脚本检查实际命令、策略参数不变、FC公式和分项求和；运行时不额外调用奖励函数。

## 目录状态

- `main_02`：正式144条轨迹、全部CSV、20张图、来源快照和诊断结论。
- `reproducibility_02`：正式同种子复跑，保留NPZ和精简统计；不是新的独立初始状态样本。
- `pilot_01`、`pilot_02`：startup DR缓冲访问接口适配失败的准备尝试，没有有效轨迹。
- `pilot_03`：6环境、3秒采集接口校验，通过；不用于性能结论。
- `main_01`、`reproducibility_01`：初步完整采集，用于检查场景构造；不是正式统计。复核时发现斜坡类型被字符串`pyramid`误匹配，且平地通道判据过窄，可能收集到走出地块后的失败。之后修正为精确地形类型匹配、按场景定义成功通道并显式截断地块越界，再以新目录完成正式实验。初步产物原样保留，没有覆盖或混入最终结果。

`diagnose.py`负责仿真/来源归档，`analyze.py`仅离线分类、导出CSV和作图，`audit.py`负责首停滞配对统计、来源链路、训练日志和同种子复跑比较。全部独立于训练和部署入口。
