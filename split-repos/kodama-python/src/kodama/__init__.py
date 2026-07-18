# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ._core import (
    core_knn,
    core_plslda,
    embedding_clustering,
    graph,
    graph_clustering,
    knncv,
    matrix as _matrix,
    matrix_graph as _matrix_graph,
    opentsne,
    pca,
    plsldacv,
    umap,
)


def _labels(labels):
    _, encoded = np.unique(np.asarray(labels), return_inverse=True)
    return (encoded + 1).astype(np.int32)


def KNNCV(data, labels, **kwargs):
    return knncv(np.asarray(data, dtype=np.float32), _labels(labels), **kwargs)


def PLSLDACV(data, labels, **kwargs):
    return plsldacv(np.asarray(data, dtype=np.float32), _labels(labels), **kwargs)


def CoreKNN(data, labels, **kwargs):
    return core_knn(np.asarray(data, dtype=np.float32), _labels(labels), **kwargs)


def CorePLSLDA(data, labels, **kwargs):
    return core_plslda(np.asarray(data, dtype=np.float32), _labels(labels), **kwargs)


def PCA(data, **kwargs):
    return pca(np.asarray(data, dtype=np.float32), **kwargs)


class KodamaMatrixResult(dict):
    """Dictionary result returned by :func:`kodama.matrix`.

    The raw C++ fields remain available by key. Convenience properties expose
    the best run, best labels, class counts, parameters, and timing table.
    """

    @property
    def class_counts(self):
        res = self.get("res")
        if res is None:
            return np.array([], dtype=np.int32)
        return np.array([len(np.unique(row)) for row in np.asarray(res)], dtype=np.int32)

    @property
    def best_run(self):
        acc = np.asarray(self.get("acc", []), dtype=float)
        if acc.size == 0 or np.all(np.isnan(acc)):
            return None
        return int(np.nanargmax(acc))

    @property
    def best_labels(self):
        run = self.best_run
        res = self.get("res")
        if run is None or res is None:
            return np.array([], dtype=np.int32)
        return np.asarray(res)[run].astype(np.int32, copy=False)

    @property
    def parameters(self):
        return self.get("parameters", {})

    def timing_table(self):
        return timing(self)

    def __repr__(self):
        run = self.best_run
        if run is None:
            best = "best_run=None"
        else:
            acc = float(np.asarray(self.get("acc"))[run])
            best = f"best_run={run + 1}, acc={acc:.4g}, classes={int(self.class_counts[run])}"
        return (
            "KodamaMatrixResult("
            f"classifier={self.get('classifier')!r}, backend={self.get('backend')!r}, "
            f"runs={self.get('parameters', {}).get('M')}, Tcycle={self.get('parameters', {}).get('Tcycle')}, "
            f"{best}, runtime_seconds={float(self.get('runtime_seconds', float('nan'))):.4g})"
        )


def _default_splitting(n_samples):
    return 100 if n_samples < 40000 else 300


def matrix(
    data,
    spatial=None,
    W=None,
    constrain=None,
    fix=None,
    M=100,
    Tcycle=20,
    ncomp=None,
    landmarks=10000,
    splitting=None,
    n_threads=4,
    graph_neighbors=100,
    knn_k=30,
    spatial_resolution=0.3,
    spatial_graph_mix=False,
    spatial_constraint_mode=0,
    metric="euclidean",
    classifier="knn",
    backend="cpu",
    seed=1234,
    progress=True,
    apply_kodama_dissimilarity=True,
):
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array")
    if ncomp is None:
        ncomp = min(50, x.shape[1])
    if splitting is None:
        splitting = _default_splitting(x.shape[0])
    spatial_array = None if spatial is None else np.asarray(spatial, dtype=np.float32)
    result = _matrix(
        x,
        spatial=spatial_array,
        W=W,
        constrain=constrain,
        fix=fix,
        M=M,
        Tcycle=Tcycle,
        ncomp=ncomp,
        landmarks=landmarks,
        splitting=splitting,
        n_threads=n_threads,
        graph_neighbors=graph_neighbors,
        knn_k=knn_k,
        spatial_resolution=spatial_resolution,
        spatial_graph_mix=spatial_graph_mix,
        spatial_constraint_mode=spatial_constraint_mode,
        metric=metric,
        classifier=classifier,
        backend=backend,
        seed=seed,
        progress=progress,
        apply_kodama_dissimilarity=apply_kodama_dissimilarity,
    )
    result = KodamaMatrixResult(result)
    result["parameters"] = {
        "M": int(M),
        "Tcycle": int(Tcycle),
        "ncomp": int(ncomp),
        "landmarks": int(landmarks),
        "splitting": int(splitting),
        "n_threads": int(n_threads),
        "graph_neighbors": int(graph_neighbors),
        "knn_k": int(knn_k),
        "spatial_resolution": float(spatial_resolution),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": int(spatial_constraint_mode),
        "metric": metric,
        "classifier": classifier,
        "backend": backend,
        "seed": int(seed),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
    }
    result["class_counts"] = result.class_counts
    best = result.best_run
    result["best_run"] = None if best is None else best + 1
    result["best_labels"] = result.best_labels
    return result


def matrix_graph(
    indices,
    distances=None,
    data=None,
    spatial=None,
    W=None,
    constrain=None,
    fix=None,
    M=100,
    Tcycle=20,
    ncomp=50,
    landmarks=10000,
    splitting=100,
    n_threads=4,
    graph_neighbors=None,
    knn_k=30,
    spatial_resolution=0.3,
    spatial_graph_mix=False,
    spatial_constraint_mode=0,
    classifier="knn",
    backend="cpu",
    graph_feature_mode="laplacian_self_tuning",
    graph_feature_components=0,
    graph_feature_steps=3,
    seed=1234,
    progress=True,
    apply_kodama_dissimilarity=True,
):
    if isinstance(indices, dict):
        distances = indices.get("distances", distances)
        indices = indices.get("indices")
    if indices is None or distances is None:
        raise ValueError("indices and distances are required")
    idx = np.asarray(indices, dtype=np.int32)
    dst = np.asarray(distances, dtype=np.float32)
    if idx.ndim != 2 or dst.ndim != 2 or idx.shape != dst.shape:
        raise ValueError("indices and distances must be matrices with the same shape")
    if graph_neighbors is None:
        graph_neighbors = idx.shape[1]
    result = _matrix_graph(
        idx,
        dst,
        data=None if data is None else np.asarray(data, dtype=np.float32),
        spatial=None if spatial is None else np.asarray(spatial, dtype=np.float32),
        W=None if W is None else np.asarray(W, dtype=np.int32),
        constrain=constrain,
        fix=fix,
        M=M,
        Tcycle=Tcycle,
        ncomp=ncomp,
        landmarks=landmarks,
        splitting=splitting,
        n_threads=n_threads,
        graph_neighbors=graph_neighbors,
        knn_k=knn_k,
        spatial_resolution=spatial_resolution,
        spatial_graph_mix=spatial_graph_mix,
        spatial_constraint_mode=spatial_constraint_mode,
        classifier=classifier,
        backend=backend,
        graph_feature_mode=graph_feature_mode,
        graph_feature_components=graph_feature_components,
        graph_feature_steps=graph_feature_steps,
        seed=seed,
        progress=progress,
        apply_kodama_dissimilarity=apply_kodama_dissimilarity,
    )
    result = KodamaMatrixResult(result)
    result["parameters"] = {
        "M": int(M),
        "Tcycle": int(Tcycle),
        "ncomp": int(ncomp),
        "landmarks": int(landmarks),
        "splitting": int(splitting),
        "n_threads": int(n_threads),
        "graph_neighbors": int(graph_neighbors),
        "knn_k": int(knn_k),
        "spatial_resolution": float(spatial_resolution),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": int(spatial_constraint_mode),
        "classifier": classifier,
        "backend": backend,
        "graph_feature_mode": graph_feature_mode,
        "graph_feature_components": int(graph_feature_components),
        "graph_feature_steps": int(graph_feature_steps),
        "graph_uses_data_geometry": data is not None,
        "seed": int(seed),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
    }
    result["class_counts"] = result.class_counts
    best = result.best_run
    result["best_run"] = None if best is None else best + 1
    result["best_labels"] = result.best_labels
    return result


def visualization(
    x,
    method="UMAP",
    init=None,
    k=30,
    metric="euclidean",
    backend="cpu",
    n_threads=4,
    gpu_device=0,
    n_epochs=200,
    n_iter=500,
    perplexity=30.0,
    graph_mode="binary",
    seed=None,
    **kwargs,
):
    if isinstance(x, dict) and "knn" in x:
        g = x["knn"]
    elif isinstance(x, dict) and "indices" in x and "distances" in x:
        g = x
    else:
        g = graph(
            np.asarray(x, dtype=np.float32),
            k=k,
            metric=metric,
            backend=backend,
            n_threads=n_threads,
            gpu_device=gpu_device,
        )

    method_l = method.lower()
    if method_l == "umap":
        return umap(
            g["indices"],
            g["distances"],
            init=init,
            n_neighbors=k,
            n_epochs=n_epochs,
            backend=backend,
            n_threads=n_threads,
            seed=1234 if seed is None else seed,
            gpu_device=gpu_device,
            graph_mode=graph_mode,
            **kwargs,
        )
    if method_l in {"t-sne", "tsne", "opentsne"}:
        return opentsne(
            g["indices"],
            g["distances"],
            init=init,
            n_neighbors=k,
            perplexity=perplexity,
            n_iter=n_iter,
            backend=backend,
            n_threads=n_threads,
            seed=4 if seed is None else seed,
            gpu_device=gpu_device,
            **kwargs,
        )
    raise ValueError(f"Unsupported visualization method: {method}")


def timing(result):
    raw = result.get("timing") if isinstance(result, dict) else None
    if raw is None and isinstance(result, dict) and "runtime_seconds" in result:
        raw = {"runtime_seconds": result["runtime_seconds"]}
    if raw is None:
        raise ValueError("No timing information found.")
    names = list(raw.keys())
    seconds = np.array([float(raw[name]) for name in names], dtype=float)
    total = raw.get("runtime_seconds", float(np.nansum(seconds)))
    total = float(total)
    percent = np.where(total > 0, 100.0 * seconds / total, np.nan)
    return [
        {"step": name, "seconds": float(sec), "percent": float(pct)}
        for name, sec, pct in zip(names, seconds, percent)
    ]


def diagnostics(all_libraries=False):
    import kodama._core as _core_module

    extension = Path(_core_module.__file__)
    linker = "otool" if platform.system() == "Darwin" else "ldd"
    args = [linker, "-L", str(extension)] if linker == "otool" else [linker, str(extension)]
    linked = []
    try:
        linked = subprocess.run(args, check=False, capture_output=True, text=True).stdout.splitlines()
    except OSError:
        linked = []
    if not all_libraries:
        keep = ("omp", "gomp", "blas", "openblas", "mkl", "cuda", "cublas", "cufft", "stdc")
        linked = [line for line in linked if any(token in line.lower() for token in keep)]
    conda = os.environ.get("CONDA_PREFIX", "")
    recommended = []
    if conda and platform.system() != "Darwin":
        for name in ("libgomp.so", "libopenblasp-r0.3.33.so", "libstdc++.so.6"):
            candidate = Path(conda) / "lib" / name
            if candidate.exists():
                recommended.append(str(candidate))
    return {
        "package": "kodama-python",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "extension": str(extension),
        "linked_libraries": linked,
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CONDA_PREFIX",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "DYLD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "recommended_ld_preload": recommended,
    }


def clustering(
    x,
    n_clusters=0,
    weight="distance",
    k=30,
    metric="euclidean",
    graph_backend="cpu",
    n_threads=4,
    n_iterations=10,
    random_walk_steps=4,
    gpu_device=0,
):
    if isinstance(x, dict) and "indices" in x and "distances" in x:
        return graph_clustering(
            x["indices"],
            x["distances"],
            weight=weight,
            n_threads=n_threads,
            n_iterations=n_iterations,
            random_walk_steps=random_walk_steps,
            n_clusters=n_clusters,
        )
    return embedding_clustering(
        np.asarray(x, dtype=np.float32),
        graph_backend=graph_backend,
        weight=weight,
        metric=metric,
        k=k,
        n_threads=n_threads,
        n_iterations=n_iterations,
        random_walk_steps=random_walk_steps,
        n_clusters=n_clusters,
        gpu_device=gpu_device,
    )


KODAMA = SimpleNamespace(
    matrix=matrix,
    matrix_graph=matrix_graph,
    visualization=visualization,
    graph=graph,
    clustering=clustering,
    timing=timing,
    diagnostics=diagnostics,
    pca=PCA,
)

KODAMA_matrix = matrix
KODAMA_matrix_graph = matrix_graph
KODAMA_visualization = visualization
KODAMA_graph = graph
KODAMA_clustering = clustering
KODAMA_timing = timing
KODAMA_diagnostics = diagnostics
makeSNNGraph = graph
KODAMA_pca = PCA

__all__ = [
    "KNNCV",
    "PLSLDACV",
    "CoreKNN",
    "CorePLSLDA",
    "PCA",
    "KODAMA",
    "KODAMA_matrix",
    "KODAMA_matrix_graph",
    "KODAMA_visualization",
    "KODAMA_graph",
    "KODAMA_clustering",
    "KODAMA_timing",
    "KODAMA_diagnostics",
    "KODAMA_pca",
    "KodamaMatrixResult",
    "makeSNNGraph",
    "clustering",
    "core_knn",
    "core_plslda",
    "diagnostics",
    "embedding_clustering",
    "graph",
    "graph_clustering",
    "knncv",
    "matrix",
    "matrix_graph",
    "opentsne",
    "pca",
    "plsldacv",
    "timing",
    "umap",
    "visualization",
]
