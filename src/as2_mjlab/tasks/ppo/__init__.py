"""Register the independent PPO baseline without legacy aliases."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import ppo_flat_env_cfg, ppo_rough_env_cfg
from .rl_cfg import ppo_runner_cfg


register_mjlab_task(
    task_id="as2-flat-ppo", env_cfg=ppo_flat_env_cfg(), play_env_cfg=ppo_flat_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg(), runner_cls=MjlabOnPolicyRunner,
)
register_mjlab_task(
    task_id="as2-rough-ppo", env_cfg=ppo_rough_env_cfg(), play_env_cfg=ppo_rough_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg(), runner_cls=MjlabOnPolicyRunner,
)
