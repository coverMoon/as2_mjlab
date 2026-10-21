"""MjLab training CLI with the task ID as the default log run name."""

from dataclasses import replace
import sys

import mjlab
import tyro
from mjlab.scripts import train as mjlab_train
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks


def launch_training(task_id: str, args: mjlab_train.TrainConfig) -> None:
    # Upstream creates <timestamp>_<run_name>. An explicit name takes priority.
    if not args.agent.run_name:
        args = replace(args, agent=replace(args.agent, run_name=task_id))
    mjlab_train.launch_training(task_id, args)


def main() -> None:
    maybe_print_top_level_help("train")
    import mjlab.tasks  # noqa: F401 -- populates the registry, including plugins.

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        mjlab_train.TrainConfig,
        args=remaining_args,
        default=mjlab_train.TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()
