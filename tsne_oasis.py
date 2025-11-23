"""t-SNE visualization for OASIS unlearning experiments.

This script reuses the trained/unlearned models and loaders from ``fu_oasis_kl``
without modifying or retraining them. It projects logits to 2D using PCA +
t-SNE and highlights samples originating from the erased party.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import fu_oasis_kl as fu


def _ensure_required_state():
    """Validate that fu_oasis_kl exposes the required experiment objects."""

    required = [
        "unlearned_model_dict",
        "testloader_clean",
        "party_to_be_erased",
    ]
    missing = [name for name in required if not hasattr(fu, name)]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "fu_oasis_kl is missing required attributes: " f"{missing_list}. "
            "Please run fu_oasis_kl to complete training before launching t-SNE."
        )


def load_model(state_dict: dict) -> torch.nn.Module:
    """Instantiate OASISNet and load the provided state dict."""

    model = fu.OASISNet().to(fu.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect logits and labels from the provided dataloader."""

    features = []
    labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(fu.device)
        logits = model(x_batch)
        features.append(logits.detach().cpu().numpy())
        labels.append(y_batch.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_test_party_labels(
    testloader_clean: torch.utils.data.DataLoader,
    indices_per_party: Sequence[Sequence[int]] | None,
    party_to_be_erased: int,
) -> np.ndarray:
    """Map each test sample to a boolean flag indicating forgotten membership."""

    dataset = testloader_clean.dataset
    if hasattr(dataset, "sample_indices"):
        test_indices: Iterable[int] = dataset.sample_indices
    else:
        test_indices = range(len(dataset))

    forget_set = set(indices_per_party[party_to_be_erased]) if indices_per_party else set()
    labels = [idx in forget_set for idx in test_indices]
    return np.array(labels, dtype=bool)


def _project_tsne(features: np.ndarray, random_state: int = 0) -> np.ndarray:
    pca = PCA(n_components=min(50, features.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(features)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=random_state,
        init="pca",
    )
    return tsne.fit_transform(reduced)


def plot_tsne(
    retrain_features: np.ndarray,
    unlearn_features: np.ndarray,
    y_test: np.ndarray,
    test_party_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Generate side-by-side t-SNE scatter plots for retrain vs unlearn models."""

    retrain_tsne = _project_tsne(retrain_features)
    unlearn_tsne = _project_tsne(unlearn_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    titles = ["Retrain", "Unlearn"]
    embeddings = [retrain_tsne, unlearn_tsne]

    markers = {True: "o", False: "^"}
    labels_map = {True: "Forgotten", False: "Retained"}

    for ax, emb, title in zip(axes, embeddings, titles):
        for forgotten in (True, False):
            mask = test_party_labels == forgotten
            scatter = ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=y_test[mask],
                cmap="tab10",
                marker=markers[forgotten],
                alpha=0.7,
                edgecolors="w",
                linewidths=0.3,
                label=labels_map[forgotten],
            )
        ax.set_title(f"t-SNE of {title} model", fontsize=14)
        ax.set_xlabel("t-SNE component 1")
        ax.set_ylabel("t-SNE component 2")
        legend = ax.legend(loc="lower left")
        legend.get_frame().set_alpha(0.8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Class label")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved t-SNE visualization to {output_path}")


def main() -> None:
    _ensure_required_state()

    indices_per_party = getattr(fu, "indices_per_party", None)

    retrain_state = fu.unlearned_model_dict.get("Retrain")
    unlearn_state = fu.unlearned_model_dict.get("Unlearn")
    if retrain_state is None or unlearn_state is None:
        raise RuntimeError(
            "unlearned_model_dict must contain 'Retrain' and 'Unlearn' checkpoints."
        )

    retrain_model = load_model(retrain_state)
    unlearn_model = load_model(unlearn_state)

    retrain_features, y_test = extract_features(retrain_model, fu.testloader_clean)
    unlearn_features, _ = extract_features(unlearn_model, fu.testloader_clean)

    test_party_labels = build_test_party_labels(
        fu.testloader_clean, indices_per_party, fu.party_to_be_erased
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).resolve().parent.parent / "doc" / "images" / f"tsne_retrain_unlearn_{timestamp}.png"
    plot_tsne(retrain_features, unlearn_features, y_test, test_party_labels, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - user-facing runtime guard
        sys.stderr.write(f"[t-SNE] Failed: {exc}\n")
        sys.exit(1)
