import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_experiment(exp_dir: Path):
    config_path = exp_dir / "config.json"
    metrics_path = exp_dir / "metrics.npz"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing config or metrics in {exp_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    metrics = np.load(metrics_path, allow_pickle=True)
    return config, metrics


def plot_accuracy_curves(exp_dir: Path, config: dict, metrics):
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rounds_phase1 = metrics["rounds_phase1"].tolist()
    rounds_phase3 = metrics["rounds_phase3"].tolist()

    retrain_phase1_clean = metrics["clean_acc_retrain_phase1"]
    retrain_phase1_bd = metrics["bd_acc_retrain_phase1"]
    fedavg_phase1_clean = metrics["clean_acc_fedavg_phase1"]
    fedavg_phase1_bd = metrics["bd_acc_fedavg_phase1"]

    retrain_phase3_clean = metrics["clean_acc_retrain_phase3"]
    retrain_phase3_bd = metrics["bd_acc_retrain_phase3"]
    unlearn_phase3_clean = metrics["clean_acc_unlearn_phase3"]
    unlearn_phase3_bd = metrics["bd_acc_unlearn_phase3"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds_phase1, retrain_phase1_clean, "m^-", label="Retrain Clean (Phase1)")
    ax.plot(rounds_phase1, retrain_phase1_bd, "c+-", label="Retrain Backdoor (Phase1)")
    ax.plot(rounds_phase1, fedavg_phase1_clean, "bo-", label="FedAvg Clean (Phase1)")
    ax.plot(rounds_phase1, fedavg_phase1_bd, "yo-", label="FedAvg Backdoor (Phase1)")

    offset = rounds_phase1[-1] + 1 if rounds_phase1 else 0
    rounds_phase3_shifted = [offset + r for r in rounds_phase3]
    ax.plot(rounds_phase3_shifted, retrain_phase3_clean, "m--", label="Retrain Clean (Phase3)")
    ax.plot(rounds_phase3_shifted, retrain_phase3_bd, "c--", label="Retrain Backdoor (Phase3)")
    ax.plot(rounds_phase3_shifted, unlearn_phase3_clean, "r--", label="Unlearn Clean (Phase3)")
    ax.plot(rounds_phase3_shifted, unlearn_phase3_bd, "g--", label="Unlearn Backdoor (Phase3)")

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"OASIS Unlearn vs Retrain (num_parties={config.get('num_parties')})")
    ax.grid(True)
    ax.legend()

    acc_path = fig_dir / f"acc_curves_np{config.get('num_parties')}.png"
    fig.tight_layout()
    fig.savefig(acc_path)
    plt.close(fig)
    return acc_path


def plot_distance_bar(exp_dir: Path, config: dict, metrics):
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stages = ["Before_Unlearn", "After_Unlearn_Phase2", "After_Unlearn_Final"]
    values = [
        float(metrics["dist_before_unlearn"]),
        float(metrics["dist_after_unlearn_phase2"]),
        float(metrics["dist_after_unlearn_final"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stages, values, marker="o")
    ax.set_ylabel("L2 Distance")
    ax.set_title(f"Model Distance Progression (num_parties={config.get('num_parties')})")
    ax.grid(True)

    dist_path = fig_dir / f"distance_np{config.get('num_parties')}.png"
    fig.tight_layout()
    fig.savefig(dist_path)
    plt.close(fig)
    return dist_path


def plot_ablation(exp_dirs):
    num_parties_list = []
    distances_final = []

    for exp_dir in exp_dirs:
        config, metrics = load_experiment(exp_dir)
        num_parties_list.append(config.get("num_parties"))
        distances_final.append(float(metrics["dist_after_unlearn_final"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(num_parties_list, distances_final, "o-", label="Final Distance")
    ax.set_xlabel("num_parties")
    ax.set_ylabel("L2 Distance (Unlearn vs Retrain)")
    ax.set_title("Distance vs num_parties")
    ax.grid(True)
    ax.legend()

    ablation_path = Path("doc/experiments/ablation_distance.png")
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ablation_path)
    plt.close(fig)
    return ablation_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot OASIS unlearning results")
    parser.add_argument(
        "--exp-dirs",
        nargs="+",
        required=True,
        help="List of experiment directories to plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dirs = [Path(p) for p in args.exp_dirs]

    for exp_dir in exp_dirs:
        config, metrics = load_experiment(exp_dir)
        acc_path = plot_accuracy_curves(exp_dir, config, metrics)
        dist_path = plot_distance_bar(exp_dir, config, metrics)
        print(f"Saved accuracy curves to {acc_path}")
        print(f"Saved distance plot to {dist_path}")

    if len(exp_dirs) > 1:
        ablation_path = plot_ablation(exp_dirs)
        print(f"Saved ablation plot to {ablation_path}")


if __name__ == "__main__":
    main()
