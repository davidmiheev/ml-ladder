"""A3. k-means and PCA from scratch, compared against each other on the same toy dataset

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

Both methods answer "what structure is in this cloud?" with the same
least-squares objective and two different answers: k-means partitions the points
and represents each by its cluster mean; PCA keeps every point and represents it
in a lower-dimensional affine subspace. Running them on one dataset makes the
contrast concrete — the reconstruction error of a k-cluster quantization against
that of a k-dimensional projection, and what each recovers of the true labels.

Run with ``ladder run A3`` (``--help`` for flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, datasets, experiment, metrics, plotting

Array = NDArray[np.float64]


@dataclass
class KMeans:
    """Lloyd's algorithm with k-means++ seeding and restarts.

    Restarts matter: Lloyd's only finds a local optimum, and the seeding is what
    decides which one. ``inertia_`` is the objective — within-cluster sum of
    squares — so the best restart is simply the smallest one.
    """

    n_clusters: int = 3
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-6
    rng: np.random.Generator | int | None = None
    centers_: Array = field(default_factory=lambda: np.zeros((0, 0)))
    labels_: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    inertia_: float = np.inf
    n_iter_: int = 0

    def _plus_plus_init(self, features: Array, generator: np.random.Generator) -> Array:
        """Seed centres far apart: each new centre is drawn ∝ squared distance."""
        centers = [features[generator.integers(len(features))]]
        for _ in range(1, self.n_clusters):
            squared = np.min(((features[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            total = squared.sum()
            if total <= 0:  # every point already coincides with a centre
                centers.append(features[generator.integers(len(features))])
                continue
            centers.append(features[generator.choice(len(features), p=squared / total)])
        return np.array(centers)

    @staticmethod
    def _assign(features: Array, centers: Array) -> tuple[NDArray[np.int64], Array]:
        """Nearest centre per point, plus the squared distance to it."""
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        return labels, distances[np.arange(len(features)), labels]

    def _lloyd(self, samples: Array, centers: Array) -> tuple[Array, NDArray[np.int64], float, int]:
        """One run of Lloyd's from the given seeding, to convergence or max_iter."""
        previous = np.inf
        iterations = 0
        while iterations < self.max_iter:
            iterations += 1
            labels, squared = self._assign(samples, centers)
            inertia = float(squared.sum())
            for cluster in range(self.n_clusters):
                members = samples[labels == cluster]
                # An emptied cluster is re-seeded at the worst-fit point.
                centers[cluster] = members.mean(axis=0) if len(members) else samples[int(np.argmax(squared))]
            if previous - inertia <= self.tol * max(previous, 1.0):
                break
            previous = inertia

        labels, squared = self._assign(samples, centers)
        return centers, labels, float(squared.sum()), iterations

    def fit(self, features: Array) -> KMeans:
        generator = datasets.as_rng(self.rng)
        samples = np.asarray(features, dtype=np.float64)
        self.inertia_ = np.inf

        for _ in range(self.n_init):
            centers, labels, inertia, iterations = self._lloyd(samples, self._plus_plus_init(samples, generator))
            if inertia < self.inertia_:
                self.inertia_, self.centers_, self.labels_, self.n_iter_ = inertia, centers.copy(), labels, iterations
        return self

    def predict(self, features: Array) -> NDArray[np.int64]:
        return self._assign(np.asarray(features, dtype=np.float64), self.centers_)[0]

    def transform(self, features: Array) -> Array:
        """Quantize: replace each point by its cluster centre."""
        return self.centers_[self.predict(features)]

    def reconstruction_error(self, features: Array) -> float:
        samples = np.asarray(features, dtype=np.float64)
        return float(np.mean(((samples - self.transform(samples)) ** 2).sum(axis=1)))


@dataclass
class PCA:
    """Principal components via the SVD of the centred data matrix.

    SVD rather than an eigendecomposition of the covariance: same answer, better
    conditioning, and the singular values give the explained variance directly.
    """

    n_components: int = 2
    mean_: Array = field(default_factory=lambda: np.zeros(0))
    components_: Array = field(default_factory=lambda: np.zeros((0, 0)))
    explained_variance_: Array = field(default_factory=lambda: np.zeros(0))
    singular_values_: Array = field(default_factory=lambda: np.zeros(0))

    def fit(self, features: Array) -> PCA:
        samples = np.asarray(features, dtype=np.float64)
        self.mean_ = samples.mean(axis=0)
        centred = samples - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
        # Sign convention: largest-magnitude loading of each component is positive.
        signs = np.sign(right_vectors[np.arange(len(right_vectors)), np.argmax(np.abs(right_vectors), axis=1)])
        right_vectors = right_vectors * signs[:, None]

        self.singular_values_ = singular_values
        self.explained_variance_ = singular_values**2 / (len(samples) - 1)
        self.components_ = right_vectors[: self.n_components]
        return self

    @property
    def explained_variance_ratio_(self) -> Array:
        total = self.explained_variance_.sum()
        return self.explained_variance_ / total if total > 0 else self.explained_variance_

    def _take(self, n_components: int | None) -> Array:
        return self.components_[: self.n_components if n_components is None else n_components]

    def transform(self, features: Array, n_components: int | None = None) -> Array:
        components = self._take(n_components)
        return (np.asarray(features, dtype=np.float64) - self.mean_) @ components.T

    def inverse_transform(self, scores: Array, n_components: int | None = None) -> Array:
        components = self._take(n_components)
        return np.asarray(scores, dtype=np.float64) @ components + self.mean_

    def reconstruction_error(self, features: Array, n_components: int | None = None) -> float:
        samples = np.asarray(features, dtype=np.float64)
        rebuilt = self.inverse_transform(self.transform(samples, n_components), n_components)
        return float(np.mean(((samples - rebuilt) ** 2).sum(axis=1)))


@experiment(
    "a3_kmeans_and_pca",
    defaults={
        "samples": 600,
        "features": 8,
        "clusters": 4,
        "cluster_std": 1.6,
        "k_values": [2, 3, 4, 5, 6, 8],
        "n_init": 10,
        "pca_components": 2,
    },
    description="k-means and PCA on one dataset: elbow, explained variance, and both reconstruction errors.",
)
def run(exp: Experiment) -> dict[str, float]:
    """k-means and PCA from scratch, compared against each other on the same toy dataset."""
    rng = np.random.default_rng(exp.seed)
    features, labels, true_centers = datasets.make_blobs(
        exp["samples"], centers=exp["clusters"], n_features=exp["features"], cluster_std=exp["cluster_std"], rng=rng
    )
    (features,) = datasets.standardize(features)

    # -- k-means: how many clusters does the data support? ----------------
    inertias, purities, rand_scores = [], [], []
    models: dict[int, KMeans] = {}
    for k in exp["k_values"]:
        model = KMeans(n_clusters=k, n_init=exp["n_init"], rng=rng).fit(features)
        models[k] = model
        purity = metrics.cluster_purity(labels, model.labels_)
        rand = metrics.adjusted_rand_index(labels, model.labels_)
        inertias.append(model.inertia_)
        purities.append(purity)
        rand_scores.append(rand)
        exp.log({"k": k, "kmeans/inertia": model.inertia_, "kmeans/purity": purity, "kmeans/ari": rand})
        print(f"[k={k}] inertia={model.inertia_:9.2f}  purity={purity:.4f}  ARI={rand:+.4f}")

    best_k = int(exp["k_values"][int(np.argmax(rand_scores))])
    kmeans = models[best_k]
    print(f"[kmeans] best k by ARI = {best_k} (true clusters = {exp['clusters']})")

    # -- PCA: how many directions does the data need? ---------------------
    pca = PCA(n_components=exp["features"]).fit(features)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    for index, (ratio, total) in enumerate(zip(ratios, cumulative, strict=True), start=1):
        exp.log({"component": index, "pca/explained_variance_ratio": float(ratio), "pca/cumulative": float(total)})
    print(
        f"[pca] top-{exp['pca_components']} components explain {cumulative[exp['pca_components'] - 1]:.1%} of variance"
    )

    # -- the comparison: same k, two kinds of compression -----------------
    # k-means with k centres vs. PCA with k components — both replace each point
    # by a least-squares approximation, one discrete, one continuous.
    kmeans_errors, pca_errors = [], []
    for k in exp["k_values"]:
        kmeans_errors.append(models[k].reconstruction_error(features))
        pca_errors.append(pca.reconstruction_error(features, n_components=min(k, exp["features"])))
        print(f"[reconstruct k={k}] k-means={kmeans_errors[-1]:.4f}  pca={pca_errors[-1]:.4f}")

    # -- clustering in the PCA subspace ------------------------------------
    projected = pca.transform(features, n_components=exp["pca_components"])
    reduced = KMeans(n_clusters=exp["clusters"], n_init=exp["n_init"], rng=rng).fit(projected)
    raw = models.get(exp["clusters"]) or KMeans(n_clusters=exp["clusters"], n_init=exp["n_init"], rng=rng).fit(features)
    raw_ari = metrics.adjusted_rand_index(labels, raw.labels_)
    reduced_ari = metrics.adjusted_rand_index(labels, reduced.labels_)
    print(f"[pca+kmeans] ARI raw={raw_ari:+.4f}  on {exp['pca_components']} components={reduced_ari:+.4f}")

    # -- figures ----------------------------------------------------------
    exp.log_figure(
        "kmeans_elbow",
        plotting.plot_curves(
            {"inertia": inertias},
            x=exp["k_values"],
            title="k-means — elbow",
            xlabel="k",
            ylabel="within-cluster SSE",
            marker="o",
        ),
    )
    exp.log_figure(
        "kmeans_agreement",
        plotting.plot_curves(
            {"purity": purities, "adjusted Rand": rand_scores},
            x=exp["k_values"],
            title="k-means — agreement with true labels",
            xlabel="k",
            ylabel="score",
            marker="o",
        ),
    )
    exp.log_figure(
        "pca_variance",
        plotting.plot_curves(
            {"per component": list(ratios), "cumulative": list(cumulative)},
            x=list(range(1, len(ratios) + 1)),
            title="PCA — explained variance",
            xlabel="component",
            ylabel="fraction of variance",
            marker="o",
        ),
    )
    exp.log_figure(
        "reconstruction_error",
        plotting.plot_curves(
            {"k-means (k centres)": kmeans_errors, "PCA (k components)": pca_errors},
            x=exp["k_values"],
            title="Same budget, two compressions",
            xlabel="k",
            ylabel="mean squared reconstruction error",
            logy=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "pca_true_labels",
        plotting.plot_scatter(
            projected,
            labels,
            title="PCA projection — true labels",
            xlabel="PC1",
            ylabel="PC2",
            centers=pca.transform(true_centers, n_components=2) if exp["features"] == true_centers.shape[1] else None,
        ),
    )
    exp.log_figure(
        "pca_kmeans_labels",
        plotting.plot_scatter(
            projected,
            kmeans.labels_,
            title=f"PCA projection — k-means labels (k={best_k})",
            xlabel="PC1",
            ylabel="PC2",
            centers=pca.transform(kmeans.centers_, n_components=2),
        ),
    )

    return {
        "kmeans/best_k_by_ari": float(best_k),
        "kmeans/best_ari": float(max(rand_scores)),
        "kmeans/purity_at_true_k": metrics.cluster_purity(labels, raw.labels_),
        "kmeans/inertia_at_true_k": raw.inertia_,
        "pca/variance_in_2d": float(cumulative[1]),
        "pca/components_for_90pct": float(int(np.searchsorted(cumulative, 0.9)) + 1),
        "ari/raw_features": raw_ari,
        "ari/pca_subspace": reduced_ari,
    }


main = run

if __name__ == "__main__":
    raise SystemExit(main())
