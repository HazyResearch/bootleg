import re
import tempfile
from pathlib import Path
from subprocess import call

import argh
from rich.console import Console

from bootleg.utils.utils import load_yaml_file

console = Console(soft_wrap=True)
bert_dir = tempfile.TemporaryDirectory().name

checkpoint_regex = re.compile(r"checkpoint_(\d+\.{0,1}\d*).model.pth")


def find_latest_checkpoint(path):
    path = Path(path)
    possible_checkpoints = []
    for fld in path.iterdir():
        res = checkpoint_regex.match(fld.name)
        if res:
            possible_checkpoints.append([res.group(1), fld])
    if len(possible_checkpoints) <= 0:
        return possible_checkpoints
    newest_sort = sorted(possible_checkpoints, key=lambda x: float(x[0]), reverse=True)
    return newest_sort[0][-1]


@argh.arg("--config", help="Path for config")
@argh.arg("--num_gpus", help="Num gpus")
@argh.arg("--batch", help="Batch size")
@argh.arg("--grad_accum", help="Grad accum")
@argh.arg("--cand_gen_run", help="Launch cand get")
def main(
    config="configs/gcp/bootleg_test.yaml",
    num_gpus=4,
    batch=None,
    grad_accum=None,
    cand_gen_run=False,
):
    config = Path(config)
    config_d = load_yaml_file(config)
    save_path = Path(config_d["emmental"]["log_path"])
    seed = config_d["emmental"].get("seed", 1234)
    call_file = "bootleg/run.py" if not cand_gen_run else "cand_gen/train.py"
    to_call = [
        "python3",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        call_file,
        "--config",
        str(config),
    ]
    # if this is a second+ run, log path will be {log_path}_{num_steps_trained}
    possible_save_paths = save_path.parent.glob(f"{save_path.name}*")
    latest_save_path = sorted(
        possible_save_paths,
        key=lambda x: int(x.name.split("_")[-1])
        if x.name.split("_")[-1].isnumeric()
        else 0,
        reverse=True,
    )
    save_path = latest_save_path[0] if len(latest_save_path) > 0 else None

    if save_path is not None and save_path.exists() and save_path.is_dir():
        last_checkpoint = find_latest_checkpoint(save_path)
        if last_checkpoint is not None:
            to_call.append("--emmental.model_path")
            to_call.append(str(save_path / last_checkpoint.name))
            num_steps_trained = int(
                checkpoint_regex.match(last_checkpoint.name).group(1)
            )
            assert num_steps_trained == int(
                float(checkpoint_regex.match(last_checkpoint.name).group(1))
            )
            optimizer_path = str(
                save_path / last_checkpoint.name.replace("model", "optimizer")
            )
            scheduler_path = str(
                save_path / last_checkpoint.name.replace("model", "scheduler")
            )
            to_call.append("--emmental.optimizer_path")
            to_call.append(optimizer_path)
            to_call.append("--emmental.scheduler_path")
            to_call.append(scheduler_path)
            to_call.append("--emmental.steps_learned")
            to_call.append(str(num_steps_trained))
            # In case didn't get through epoch, change seed so that data is reshuffled
            to_call.append("--emmental.seed")
            to_call.append(str(seed + num_steps_trained))
            to_call.append("--emmental.log_path")
            to_call.append(
                str(save_path.parent / f"{save_path.name}_{num_steps_trained}")
            )
    if batch is not None:
        to_call.append("--train_config.batch_size")
        to_call.append(str(batch))
    if grad_accum is not None:
        to_call.append("--emmental.gradient_accumulation_steps")
        to_call.append(str(grad_accum))
    print(f"CALLING...{' '.join(to_call)}")
    call(to_call)


if __name__ == "__main__":
    argh.dispatch_command(main)
