from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import as2_flat_env_cfg, as2_rough_env_cfg
from .rl_cfg import as2_ppo_runner_cfg
from .runner import AS2OnPolicyRunner


register_mjlab_task(
    task_id="as2-flat-unitree",
    env_cfg=as2_flat_env_cfg(),
    play_env_cfg=as2_flat_env_cfg(play=True),
    rl_cfg=as2_ppo_runner_cfg(),
    runner_cls=AS2OnPolicyRunner,
)

register_mjlab_task(
    task_id="as2-rough-unitree",
    env_cfg=as2_rough_env_cfg(),
    play_env_cfg=as2_rough_env_cfg(play=True),
    rl_cfg=as2_ppo_runner_cfg(),
    runner_cls=AS2OnPolicyRunner,
)
