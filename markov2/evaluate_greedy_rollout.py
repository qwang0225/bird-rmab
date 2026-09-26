"""Evaluate observation-greedy and model-informed activation-greedy."""
from pathlib import Path
import numpy as np
from env import MarkovRMABConfig, MarkovRMABEnv
from baselines import GreedyPolicy, evaluate_policy
from run import plot_results


class ModelPolicy:
    def __init__(self, transition_prob: np.ndarray, K: int):
        self.prob = transition_prob  # [arm, state, action] -> P(next state = 1)
        self.N, _, _ = self.prob.shape
        self.K = K

    def reset(self):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        idx = np.arange(self.N)
        score = self.prob[idx, state, 1] - self.prob[idx, state, 0]
        action = np.zeros(self.N, dtype=np.int32)
        action[np.argsort(-score, kind="stable")[:self.K]] = 1
        return action


def main():
    root = Path(__file__).resolve().parent
    with np.load(root / "results_50_10.npz") as saved:
        assert len(saved["random"]) == 100
        results = {name: saved[name].copy() for name in saved.files}
    cfg = MarkovRMABConfig(N=50, K=10, T=100, seed_params=0)
    env = MarkovRMABEnv(cfg, seed=0)
    model = np.stack([
        np.stack([env.q0, env.p0], axis=-1),
        np.stack([env.q1, env.p1], axis=-1),
    ], axis=1)
    results["obs_greedy"] = evaluate_policy(GreedyPolicy(cfg.N, cfg.K), cfg, 100, seed_offset=500)
    assert np.array_equal(results["obs_greedy"], results["greedy"])
    results["activation_greedy"] = evaluate_policy(ModelPolicy(model, cfg.K), cfg, 100, seed_offset=500)
    del results["greedy"]
    names = ["random", "obs_greedy", "activation_greedy",
             "ppo", "neurwin", "dpmd", "wiql", "true_whittle"]
    results = {name: results[name] for name in names}
    output = root / "results_50_10_known_activation.npz"
    np.savez(output, **results, activation_uses_true_transitions=np.asarray(True))
    plot_results(results, cfg, save_path=str(root / "comparison_50_10_known_activation.png"))
    rand = results["random"].mean()
    oracle = results["true_whittle"].mean()
    for name, values in results.items():
        gap = 100 * (values.mean() - rand) / (oracle - rand)
        print(f"{name:20s} {values.mean():8.1f} +/- {values.std():6.1f}  gap={gap:+6.1f}%", flush=True)
    print(output)


if __name__ == "__main__":
    main()
