"""Check saved policy weights before running evaluation batches."""
from pathlib import Path
import argparse
import sys


ENVIRONMENTS = ("synthetic-stationary", "synthetic-drifting", "mimic-icu")
POLICIES = ("dpmd", "neurwin", "ppo", "mlp_actor", "gaussian_actor", "learned_rollout")


def required_paths(scope):
    if scope in ("all", "main", "scale", "timing"):
        policies = ("dpmd",) if scope == "timing" else POLICIES
        for environment in ENVIRONMENTS:
            for policy in policies:
                yield Path(environment) / f"checkpoints_{policy}/best.pth"
    if scope in ("all", "main"):
        for policy in ("dpmd", "neurwin", "ppo"):
            yield Path("markov2") / f"checkpoints_{policy}/best.pth"


def validate(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("expected a nonempty checkpoint dictionary")
    tensors = []

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(checkpoint)
    if not tensors or not any(t.numel() for t in tensors):
        raise ValueError("no saved model tensors")
    if any(not torch.isfinite(t).all().item() for t in tensors):
        raise ValueError("checkpoint contains non-finite tensors")
    if path.parent.parent.name == "mimic-icu" and path.parent.name.startswith("checkpoints_ppo"):
        if not all(key in checkpoint for key in ("encoder", "actor", "critic")):
            raise ValueError("PPO requires shared per-arm MLP weights; retrain the observation-only checkpoint")
        saved = checkpoint.get("cfg", {})
        weight = checkpoint["encoder"].get("net.0.weight")
        if saved.get("obs_dim") != 5 or weight is None or weight.ndim != 2 or weight.shape[1] != 6 * saved.get("L", 0):
            raise ValueError("expected shared per-arm MLP PPO with five observations per arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scope", choices=("all", "main", "scale", "timing"), default="all")
    args = parser.parse_args()
    failures = 0
    paths = list(required_paths(args.scope))
    for relative in paths:
        path = args.root / relative
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("missing or empty; train this policy first")
            validate(path)
            print(f"[OK] {relative}")
        except Exception as error:
            failures += 1
            print(f"[ERROR] {relative}: {error}")
    if failures:
        print(f"Evaluation blocked: {failures}/{len(paths)} checkpoints need attention.")
        return 1
    print(f"All {len(paths)} checkpoint files passed validation.")
    print("This checks saved weights, not completed training epochs or convergence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
