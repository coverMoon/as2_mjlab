"""Flat / Rough 共用的非对称 Actor-Critic PPO 训练配置。"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """构造独立 PPO 配置；网络和优化超参数与原有基线保持一致。"""

    # ------------------------------------------------------------------
    # 网络：Actor 使用本体观测，Critic 使用特权观测，输入由 env_cfg 定义。
    # 两者分别维护观测归一化统计量，并随 checkpoint 保存。
    # ------------------------------------------------------------------
    actor = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,  # 训练时动作探索的初始标准差。
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )

    # ------------------------------------------------------------------
    # PPO 优化：损失、数据复用、学习率和回报估计。
    # ------------------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        # 策略更新使用 PPO clip；价值网络同时启用 value loss clip。
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,

        # 每批 rollout 数据重复优化 5 轮，每轮分为 4 个 mini-batch。
        num_learning_epochs=5,
        num_mini_batches=4,

        # 根据 KL 与目标的差异自适应调整学习率，并裁剪梯度范数。
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,

        # 折扣因子与 GAE 参数，用于估计回报和优势。
        gamma=0.99,
        lam=0.95,
    )

    # ------------------------------------------------------------------
    # 采样与保存：一次 iteration 的样本量 = 环境数量 * num_steps_per_env。
    # env_cfg 中的指令课程也使用这个默认采样长度换算阶段边界。
    # ------------------------------------------------------------------
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        experiment_name="as2_ppo",
        num_steps_per_env=24,
        max_iterations=10001,
        save_interval=100,
    )
