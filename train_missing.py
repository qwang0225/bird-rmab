"""Train missing baseline checkpoints, then validate weights before evaluation."""
from pathlib import Path
import argparse
import subprocess
import sys

from check_checkpoints import required_paths, validate


def training_command(relative):
    environment, directory, _ = relative.parts
    policy = directory.removeprefix("checkpoints_")
    scripts = {"dpmd": "diffusion_DPMD_train.py", "neurwin": "neurwin.py",
               "ppo": "ppo.py", "mlp_actor": "mlp_actor.py",
               "gaussian_actor": "gaussian_actor.py", "learned_rollout": "learned_rollout.py"}
    if environment == "mimic-icu" and policy in ("neurwin", "ppo"):
        scripts[policy] = f"train_{policy}.py"
    # Markov BIRD's entry point uses DPMDTrainConfig directly (N=50, K=10).
    if environment == "markov2" and policy == "dpmd":
        return [sys.executable, scripts[policy]]
    output_flag = "--save_dir" if policy == "neurwin" or (environment == "mimic-icu" and policy == "ppo") else "--ckpt_dir"
    n, k = (50, 10) if environment == "markov2" else (20, 5)
    return [sys.executable, scripts[policy], "--N", str(n), "--K", str(k),
            "--T", "100", output_flag, directory]


def ensure_checkpoints(root, scope="all", dry_run=False):
    paths = list(required_paths(scope))
    missing = []
    # Check every existing checkpoint before launching any expensive training.
    for relative in paths:
        path = root / relative
        if path.exists():
            validate(path)
            print(f"[REUSE] {relative}", flush=True)
        else:
            command = training_command(relative)
            cwd = root / relative.parts[0]
            if not (cwd / command[1]).is_file():
                raise FileNotFoundError(cwd / command[1])
            missing.append((relative, command, cwd))
    for relative, command, cwd in missing:
        print(f"[TRAIN] {relative}\n  {subprocess.list2cmdline(command)}", flush=True)
        if dry_run:
            continue
        subprocess.run(command, cwd=cwd, check=True)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Training finished without producing {path}")
        validate(path)
        print(f"[READY] {relative}", flush=True)
    if dry_run:
        print(f"Plan: train {len(missing)} missing checkpoints; reuse {len(paths) - len(missing)}.")
    else:
        print(f"All {len(paths)} required checkpoints are ready for evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scope", choices=("all", "main", "scale", "timing"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print missing-policy training commands without training.")
    args = parser.parse_args()
    try:
        ensure_checkpoints(args.root.resolve(), args.scope, args.dry_run)
    except Exception as error:
        print(f"[ERROR] Training/evaluation stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
