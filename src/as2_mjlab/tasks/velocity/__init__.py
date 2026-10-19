from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import as2_flat_env_cfg
from .rl_cfg import as2_ppo_runner_cfg


register_mjlab_task(
    task_id="as2-flat",
    env_cfg=as2_flat_env_cfg(),
    play_env_cfg=as2_flat_env_cfg(play=True),
    rl_cfg=as2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)