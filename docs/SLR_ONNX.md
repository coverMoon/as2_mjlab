# SLR 策略 ONNX 导出

Flat / Rough 使用现有 `export` 入口，无需启动 Viewer。默认在 checkpoint
同目录生成同名 `.onnx`，可通过 `--output-file` 指定路径。CPU 导出可避免占用训练显存：

```bash
uv run export as2-flat-slr --checkpoint-file <checkpoint.pt> --device cpu
uv run export as2-rough-slr --checkpoint-file <checkpoint.pt> --output-file /tmp/slr.onnx --device cpu
```

导出采用 checkpoint 中的 `slr_model_cfg` 网络尺寸，依据权重恢复是否启用归一化，
并严格加载模型权重。
ONNX 包含采样该 checkpoint 策略时使用的归一化统计、Encoder 和 Actor；输出为
确定性动作均值。Critic、Transition、Gaussian 探索噪声及优化器不进入图。
训练 Runner 仍只保存 checkpoint，不自动导出。

| Tensor | dtype | 默认形状 | 含义 |
|---|---|---|---|
| `history` | float32 | `[1,10,45]` | 从旧到新的 10 帧本体观测 |
| `actions` | float32 | `[1,12]` | 关节位置动作的原始均值 |

采用上游 Runner 的 opset 18、固定 batch=1 导出接口。当前帧取 `history[:, -1, :]`。
单帧顺序为：角速度 3、投影重力 3、速度指令 3、相对关节位置 12、关节速度 12、
上一步动作 12。输入应已完成环境观测项的 clip/scale；特征归一化包含在 ONNX 内，
调用端不要再次归一化。

历史缓冲区由调用端维护，不在 ONNX 内：每个控制步移除最旧一帧，追加当前帧；
初始化或 Reset 后用第一帧填满窗口，不能保留上一 Episode 的历史。
模型输出仍需按环境动作配置映射为关节目标位置：`offset + scale * actions`，
并执行实际配置的目标位置裁剪。

每次 CLI 导出会先用独立的 ONNX ReferenceEvaluator 对比 PyTorch 策略，覆盖环境
初始化历史、零输入、随机历史、放大随机历史和重复单帧历史。误差要求
`rtol=1e-4, atol=1e-5`，通过后才替换目标 ONNX。

ONNX metadata 保存 `slr_input_contract`（JSON）、任务、checkpoint 名称、控制周期、
关节名称、动作 scale/offset 及验证结果。其中观测和动作配置来自当前导出环境，
checkpoint 未存档完整训练环境契约。
现有 Manifest v1 不支持历史输入，SLR 导出不生成 `.manifest.yaml`；PPO / Unitree
的 ONNX 与 Manifest 导出行为保持原样。
