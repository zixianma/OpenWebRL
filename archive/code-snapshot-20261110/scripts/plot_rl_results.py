#!/usr/bin/env python3
"""Plot the local-browser GPT-4.1 full-300 RL evaluation curve."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITERATIONS = [10, 20, 21, 22, 30, 38, 39, 40, 50, 52, 58, 60, 69, 70, 80, 90]
SUCCESSES = [70, 95, 89, 86, 96, 107, 98, 100, 105, 92, 109, 105, 103, 103, 114, 101]
VALID = [234, 232, 233, 236, 248, 228, 228, 231, 234, 227, 230, 230, 221, 229, 229, 222]
STEALTH_O4_T06_ITERATIONS = [58, 80, 90]
STEALTH_O4_T06_SUCCESSES = [178, 169, 171]
STEALTH_O4_T06_VALID = [297, 296, 296]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    output = repo / "openwebrl" / "docs" / "rl_results" / "local_browser_gpt41_full300.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    overall = [100 * x / 300 for x in SUCCESSES]
    valid_only = [100 * x / y for x, y in zip(SUCCESSES, VALID)]
    stealth_o4_t06_overall = [100 * x / 300 for x in STEALTH_O4_T06_SUCCESSES]
    stealth_o4_t06_valid_only = [100 * x / y for x, y in zip(STEALTH_O4_T06_SUCCESSES, STEALTH_O4_T06_VALID)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax.plot(ITERATIONS, overall, marker="o", linewidth=2.2, markersize=5,
            color="#2563eb", label="Overall success")
    ax.plot(ITERATIONS, valid_only, marker="s", linewidth=2.0, markersize=4.5,
            color="#dc2626", label="Valid-only success")
    # Only three stealth T=0.6 checkpoints are available; use points rather
    # than connecting them to avoid implying an unobserved trajectory.
    ax.scatter(STEALTH_O4_T06_ITERATIONS, stealth_o4_t06_overall, marker="D", s=52,
               color="#ea580c", label="Stealth o4-mini T=0.6 overall", zorder=4)
    ax.scatter(STEALTH_O4_T06_ITERATIONS, stealth_o4_t06_valid_only, marker="P", s=58,
               color="#0891b2", label="Stealth o4-mini T=0.6 valid-only", zorder=4)
    ax.set_xlabel("Checkpoint after training iteration")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("OpenWebRL browser evaluation\nGPT-4.1 / o4-mini · 300 Online-Mind2Web tasks")
    ax.set_ylim(20, 65)
    ax.set_xlim(8, 92)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.legend(frameon=True, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
