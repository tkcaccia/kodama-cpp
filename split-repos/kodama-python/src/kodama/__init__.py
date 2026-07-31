# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import os
import platform
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ._core import (
    core_knn as _core_knn,
    core_plslda as _core_plslda,
    embedding_clustering as _embedding_clustering,
    graph as _graph,
    graph_clustering as _graph_clustering,
    knncv as _knncv,
    matrix as _matrix,
    matrix_graph as _matrix_graph,
    opentsne as _opentsne,
    pca as _pca,
    plsldacv as _plsldacv,
    umap as _umap,
    visual_init as _visual_init,
)

_BACKENDS = ("cpu", "cuda", "metal")
_VISUALIZATION_BACKENDS = ("cpu", "cuda")
_METRICS = ("euclidean", "cosine", "inner_product")
_CLASSIFIERS = ("knn", "pls_lda")
_SPATIAL_CONSTRAINT_MODES = ("kmeans", "graph", "auto")
_GRAPH_FEATURE_MODES = ("laplacian_self_tuning",)
_GRAPH_WEIGHTS = ("distance", "snn", "adaptive", "binary")


def _labels(labels):
    _, encoded = np.unique(np.asarray(labels), return_inverse=True)
    return (encoded + 1).astype(np.int32)


def _matrix32(data, name="data"):
    value = np.asarray(data, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return value


def _int_vector(value):
    return None if value is None else np.asarray(value, dtype=np.int32)


def _choice(value, name, choices):
    if value not in choices:
        options = ", ".join(repr(option) for option in choices)
        raise ValueError(f"{name} must be one of {options}")
    return value


def _spatial_constraint_code(mode):
    mode = _choice(
        mode, "spatial_constraint_mode", _SPATIAL_CONSTRAINT_MODES
    )
    return {"auto": -1, "kmeans": 0, "graph": 1}[mode]


def KNNCV(
    data,
    labels,
    constrain=None,
    folds=10,
    stratified=True,
    seed=1,
    k=10,
    metric="cosine",
    backend="cpu",
    n_cores=1,
    gpu_device=0,
):
    """Cross-validated KNN classification, mirroring ``kodamaR::KNNCV``."""
    metric = _choice(metric, "metric", ("cosine", "inner_product", "euclidean"))
    backend = _choice(backend, "backend", _BACKENDS)
    return _knncv(
        _matrix32(data),
        _labels(labels),
        constrain=_int_vector(constrain),
        folds=int(folds),
        stratified=bool(stratified),
        seed=int(seed),
        k=int(k),
        metric=metric,
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def PLSLDACV(
    data,
    labels,
    constrain=None,
    folds=10,
    stratified=True,
    seed=1,
    ncomp=None,
    center=True,
    scale=True,
    backend="cpu",
    n_cores=1,
    gpu_device=0,
):
    """Cross-validated SIMPLS plus LDA, mirroring ``kodamaR::PLSLDACV``."""
    x = _matrix32(data)
    if ncomp is None:
        ncomp = min(50, x.shape[1])
    backend = _choice(backend, "backend", _BACKENDS)
    return _plsldacv(
        x,
        _labels(labels),
        constrain=_int_vector(constrain),
        folds=int(folds),
        stratified=bool(stratified),
        seed=int(seed),
        ncomp=int(ncomp),
        center=bool(center),
        scale=bool(scale),
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def CoreKNN(
    data,
    labels,
    constrain=None,
    fix=None,
    cycles=100,
    folds=10,
    stratified=True,
    seed=1,
    k=30,
    metric="euclidean",
    backend="cpu",
    n_cores=4,
    gpu_device=0,
):
    """KODAMA label evolution with KNN, mirroring ``kodamaR::CoreKNN``."""
    metric = _choice(metric, "metric", _METRICS)
    backend = _choice(backend, "backend", _BACKENDS)
    return _core_knn(
        _matrix32(data),
        _labels(labels),
        constrain=_int_vector(constrain),
        fix=_int_vector(fix),
        cycles=int(cycles),
        folds=int(folds),
        stratified=bool(stratified),
        seed=int(seed),
        k=int(k),
        metric=metric,
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def CorePLSLDA(
    data,
    labels,
    constrain=None,
    fix=None,
    cycles=100,
    folds=10,
    stratified=True,
    seed=1,
    ncomp=None,
    backend="cpu",
    n_cores=4,
    gpu_device=0,
):
    """KODAMA label evolution with PLS-LDA, mirroring ``kodamaR::CorePLSLDA``."""
    x = _matrix32(data)
    if ncomp is None:
        ncomp = min(50, x.shape[1])
    backend = _choice(backend, "backend", _BACKENDS)
    return _core_plslda(
        x,
        _labels(labels),
        constrain=_int_vector(constrain),
        fix=_int_vector(fix),
        cycles=int(cycles),
        folds=int(folds),
        stratified=bool(stratified),
        seed=int(seed),
        ncomp=int(ncomp),
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def kodama_pca(
    data,
    ncomp=2,
    center=True,
    scale=False,
    backend="cpu",
    n_cores=1,
    gpu_device=0,
    seed=4,
    oversample=None,
    power=None,
):
    """Backend-native float32 PCA, mirroring ``kodamaR::kodama_pca``."""
    backend = _choice(backend, "backend", _BACKENDS)
    return _pca(
        _matrix32(data),
        ncomp=int(ncomp),
        center=bool(center),
        scale=bool(scale),
        backend=backend,
        seed=int(seed),
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
        oversample=-1 if oversample is None else int(oversample),
        power=-1 if power is None else int(power),
    )


PCA = kodama_pca


def _visual_initialization(data, backend="cpu", seed=4, n_cores=1, gpu_device=0):
    """Build backend-native float32 PCA starts for UMAP and openTSNE."""
    backend = _choice(backend, "backend", _BACKENDS)
    return _visual_init(
        _matrix32(data),
        backend=backend,
        seed=int(seed),
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


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
        return int(np.nanargmax(acc)) + 1

    @property
    def best_labels(self):
        run = self.best_run
        res = self.get("res")
        if run is None or res is None:
            return np.array([], dtype=np.int32)
        return np.asarray(res)[run - 1].astype(np.int32, copy=False)

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
            acc = float(np.asarray(self.get("acc"))[run - 1])
            best = (
                f"best_run={run}, acc={acc:.4g}, "
                f"classes={int(self.class_counts[run - 1])}"
            )
        return (
            "KodamaMatrixResult("
            f"classifier={self.get('classifier')!r}, backend={self.get('backend')!r}, "
            f"runs={self.get('parameters', {}).get('M')}, Tcycle={self.get('parameters', {}).get('Tcycle')}, "
            f"{best}, runtime_seconds={float(self.get('runtime_seconds', float('nan'))):.4g})"
        )


class KodamaEmbedding(np.ndarray):
    """Float32 embedding with observable initialization provenance."""

    def __new__(cls, values, initialization, initialization_backend):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.initialization = initialization
        obj.initialization_backend = initialization_backend
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self.initialization = getattr(source, "initialization", None)
        self.initialization_backend = getattr(
            source, "initialization_backend", None
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
    n_cores=4,
    graph_neighbors=None,
    knn_k=30,
    spatial_resolution=0.4,
    spatial_graph_mix=False,
    spatial_constraint_mode="kmeans",
    metric="euclidean",
    classifier="knn",
    backend=None,
    seed=1234,
    raw_data=None,
    visual_init=True,
    progress=True,
    apply_kodama_dissimilarity=True,
):
    """Run KODAMA matrix optimization, mirroring ``kodamaR::kodama_matrix``.

    The native call builds one full-data graph before all ``M`` runs. With
    ``visual_init=True`` it also performs one float32 PCA and stores both UMAP
    and openTSNE starts for :func:`visualization`.
    """
    supplied_graph = _extract_graph(data)
    if supplied_graph is not None:
        raw = raw_data
        samples = np.asarray(supplied_graph["indices"]).shape[0]
        if ncomp is None:
            ncomp = 50 if raw is None else min(50, np.asarray(raw).shape[1])
        if splitting is None:
            splitting = _default_splitting(samples)
        if graph_neighbors is None:
            graph_neighbors = np.asarray(supplied_graph["indices"]).shape[1]
        if backend is None:
            backend = data.get("backend", "cpu")
        return matrix_graph(
            data,
            data=raw,
            spatial=spatial,
            W=W,
            constrain=constrain,
            fix=fix,
            M=M,
            Tcycle=Tcycle,
            ncomp=ncomp,
            landmarks=landmarks,
            splitting=splitting,
            n_cores=n_cores,
            graph_neighbors=graph_neighbors,
            knn_k=knn_k,
            spatial_resolution=spatial_resolution,
            spatial_graph_mix=spatial_graph_mix,
            spatial_constraint_mode=spatial_constraint_mode,
            classifier=classifier,
            backend=backend,
            seed=seed,
            visual_init=visual_init,
            progress=progress,
            apply_kodama_dissimilarity=apply_kodama_dissimilarity,
        )

    x = _matrix32(data)
    if ncomp is None:
        ncomp = min(50, x.shape[1])
    if splitting is None:
        splitting = _default_splitting(x.shape[0])
    if graph_neighbors is None:
        graph_neighbors = 100
    classifier = _choice(classifier, "classifier", _CLASSIFIERS)
    backend = _choice("cpu" if backend is None else backend, "backend", _BACKENDS)
    metric = _choice(metric, "metric", _METRICS)
    spatial_constraint_code = _spatial_constraint_code(spatial_constraint_mode)
    spatial_array = None if spatial is None else _matrix32(spatial, "spatial")
    result = _matrix(
        x,
        spatial=spatial_array,
        W=_int_vector(W),
        constrain=_int_vector(constrain),
        fix=_int_vector(fix),
        M=int(M),
        Tcycle=int(Tcycle),
        ncomp=int(ncomp),
        landmarks=int(landmarks),
        splitting=int(splitting),
        n_threads=int(n_cores),
        graph_neighbors=int(graph_neighbors),
        knn_k=int(knn_k),
        spatial_resolution=float(spatial_resolution),
        spatial_graph_mix=bool(spatial_graph_mix),
        spatial_constraint_mode=spatial_constraint_code,
        metric=metric,
        classifier=classifier,
        backend=backend,
        seed=int(seed),
        progress=bool(progress),
        apply_kodama_dissimilarity=bool(apply_kodama_dissimilarity),
        compute_visual_init=bool(visual_init),
    )
    result = KodamaMatrixResult(result)
    result["parameters"] = {
        "M": int(M),
        "Tcycle": int(Tcycle),
        "ncomp": int(ncomp),
        "landmarks": int(landmarks),
        "splitting": int(splitting),
        "n_cores": int(n_cores),
        "graph_neighbors": int(graph_neighbors),
        "knn_k": int(knn_k),
        "spatial_resolution": float(spatial_resolution),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": spatial_constraint_mode,
        "metric": metric,
        "classifier": classifier,
        "backend": backend,
        "seed": int(seed),
        "visual_init": bool(visual_init),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
    }
    result["class_counts"] = result.class_counts
    result["best_run"] = result.best_run
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
    ncomp=None,
    landmarks=10000,
    splitting=None,
    n_cores=4,
    graph_neighbors=None,
    knn_k=30,
    spatial_resolution=0.4,
    spatial_graph_mix=False,
    spatial_constraint_mode="kmeans",
    classifier="knn",
    backend=None,
    graph_feature_mode="laplacian_self_tuning",
    graph_feature_components=0,
    graph_feature_steps=3,
    seed=1234,
    visual_init=True,
    progress=True,
    apply_kodama_dissimilarity=True,
):
    """Run KODAMA from a KNN matrix, mirroring ``kodamaR::kodama_matrix_graph``."""
    graph_object = indices if isinstance(indices, dict) else None
    supplied_graph = _extract_graph(indices)
    if supplied_graph is not None:
        distances = supplied_graph.get("distances", distances)
        indices = supplied_graph.get("indices")
    if indices is None or distances is None:
        raise ValueError("indices and distances are required")
    idx = np.asarray(indices, dtype=np.int32)
    dst = np.asarray(distances, dtype=np.float32)
    if idx.ndim != 2 or dst.ndim != 2 or idx.shape != dst.shape:
        raise ValueError("indices and distances must be matrices with the same shape")
    if graph_neighbors is None:
        graph_neighbors = idx.shape[1]
    if ncomp is None:
        ncomp = 50 if data is None else min(50, np.asarray(data).shape[1])
    if splitting is None:
        splitting = _default_splitting(idx.shape[0])
    classifier = _choice(classifier, "classifier", _CLASSIFIERS)
    if backend is None:
        backend = graph_object.get("backend", "cpu") if isinstance(graph_object, dict) else "cpu"
    backend = _choice(backend, "backend", _BACKENDS)
    graph_feature_mode = _choice(
        graph_feature_mode, "graph_feature_mode", _GRAPH_FEATURE_MODES
    )
    spatial_constraint_code = _spatial_constraint_code(spatial_constraint_mode)
    result = _matrix_graph(
        idx,
        dst,
        data=None if data is None else _matrix32(data),
        spatial=None if spatial is None else _matrix32(spatial, "spatial"),
        W=_int_vector(W),
        constrain=_int_vector(constrain),
        fix=_int_vector(fix),
        M=int(M),
        Tcycle=int(Tcycle),
        ncomp=int(ncomp),
        landmarks=int(landmarks),
        splitting=int(splitting),
        n_threads=int(n_cores),
        graph_neighbors=int(graph_neighbors),
        knn_k=int(knn_k),
        spatial_resolution=float(spatial_resolution),
        spatial_graph_mix=bool(spatial_graph_mix),
        spatial_constraint_mode=spatial_constraint_code,
        classifier=classifier,
        backend=backend,
        graph_feature_mode=graph_feature_mode,
        graph_feature_components=int(graph_feature_components),
        graph_feature_steps=int(graph_feature_steps),
        seed=int(seed),
        progress=bool(progress),
        apply_kodama_dissimilarity=bool(apply_kodama_dissimilarity),
    )
    result = KodamaMatrixResult(result)
    if visual_init and isinstance(graph_object, dict):
        stored_init = graph_object.get("visual_init")
        if isinstance(stored_init, dict) and stored_init.get("backend") == backend:
            result["visual_init"] = stored_init
        elif data is not None:
            result["visual_init"] = _visual_initialization(
                data,
                backend=backend,
                seed=seed,
                n_cores=n_cores,
            )
    result["parameters"] = {
        "M": int(M),
        "Tcycle": int(Tcycle),
        "ncomp": int(ncomp),
        "landmarks": int(landmarks),
        "splitting": int(splitting),
        "n_cores": int(n_cores),
        "graph_neighbors": int(graph_neighbors),
        "knn_k": int(knn_k),
        "spatial_resolution": float(spatial_resolution),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": spatial_constraint_mode,
        "classifier": classifier,
        "backend": backend,
        "graph_feature_mode": graph_feature_mode,
        "graph_feature_components": int(graph_feature_components),
        "graph_feature_steps": int(graph_feature_steps),
        "graph_uses_data_geometry": data is not None,
        "seed": int(seed),
        "visual_init": bool(visual_init),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
    }
    result["class_counts"] = result.class_counts
    result["best_run"] = result.best_run
    result["best_labels"] = result.best_labels
    return result


def graph(
    data,
    k=100,
    metric="euclidean",
    backend="cpu",
    n_cores=4,
    gpu_device=0,
    seed=1234,
):
    """Build a reusable KODAMA graph and backend-specific PCA starts."""
    metric = _choice(metric, "metric", _METRICS)
    backend = _choice(backend, "backend", _BACKENDS)
    x = _matrix32(data)
    result = _graph(
        x,
        k=int(k),
        metric=metric,
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
        seed=int(seed),
    )
    result["parameters"] = {
        "k": int(k),
        "metric": metric,
        "backend": backend,
        "n_cores": int(n_cores),
        "gpu_device": int(gpu_device),
        "seed": int(seed),
    }
    return result


def _extract_graph(value):
    if isinstance(value, dict) and "knn" in value:
        return value["knn"]
    if (
        isinstance(value, dict)
        and "indices" in value
        and "distances" in value
    ):
        return value
    return None


def visualization(
    x,
    method="UMAP",
    init=None,
    raw_data=None,
    initialize_from_raw=True,
    k=30,
    metric="euclidean",
    backend="cpu",
    n_cores=4,
    gpu_device=0,
    n_epochs=200,
    n_iter=500,
    perplexity=30.0,
    graph_mode="fuzzy",
    seed=4,
    **kwargs,
):
    """Visualize data or a KODAMA graph, mirroring ``KODAMA.visualization``."""
    method = _choice(method, "method", ("UMAP", "t-SNE", "opentsne"))
    metric = _choice(metric, "metric", _METRICS)
    backend = _choice(backend, "backend", _VISUALIZATION_BACKENDS)
    graph_mode = _choice(graph_mode, "graph_mode", ("fuzzy", "binary"))
    raw = None
    if raw_data is not None:
        raw = _matrix32(raw_data, "raw_data")
    elif not isinstance(x, dict):
        raw = _matrix32(x)

    g = _extract_graph(x)
    if g is None:
        g = graph(
            x,
            k=k,
            metric=metric,
            backend=backend,
            n_cores=n_cores,
            gpu_device=gpu_device,
        )

    method_l = method.lower()
    method_key = "umap" if method_l == "umap" else "opentsne"
    initialization = "explicit" if init is not None else None
    initialization_backend = backend if init is not None else None
    if init is None and initialize_from_raw:
        stored = x.get("visual_init") if isinstance(x, dict) else None
        if isinstance(stored, dict) and stored.get("backend") == backend:
            init = stored.get(method_key)
            initialization = "raw_pca"
            initialization_backend = backend
        elif raw is not None:
            generated = _visual_initialization(
                raw,
                backend=backend,
                seed=seed,
                n_cores=n_cores,
                gpu_device=gpu_device,
            )
            init = generated[method_key]
            initialization = "raw_pca"
            initialization_backend = generated["backend"]
        elif isinstance(stored, dict):
            warnings.warn(
                "The stored raw-data PCA initialization uses backend "
                f"{stored.get('backend')!r}, not {backend!r}. Pass raw_data "
                "to recompute it; using the graph-only fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

    if method_l == "umap":
        values = _umap(
            g["indices"],
            g["distances"],
            init=init,
            n_neighbors=k,
            n_epochs=n_epochs,
            backend=backend,
            n_threads=n_cores,
            seed=seed,
            gpu_device=gpu_device,
            graph_mode=graph_mode,
            **kwargs,
        )
        return KodamaEmbedding(
            values,
            initialization or "graph_spectral",
            initialization_backend or "cpu",
        )
    if method_l in {"t-sne", "opentsne"}:
        values = _opentsne(
            g["indices"],
            g["distances"],
            init=init,
            n_neighbors=k,
            perplexity=perplexity,
            n_iter=n_iter,
            backend=backend,
            n_threads=n_cores,
            seed=seed,
            gpu_device=gpu_device,
            **kwargs,
        )
        return KodamaEmbedding(
            values,
            initialization or "random",
            initialization_backend or "cpu",
        )
    raise ValueError(f"Unsupported visualization method: {method}")


def timing(x):
    """Return KODAMA timing records, mirroring ``KODAMA.timing``."""
    raw = x.get("timing") if isinstance(x, dict) else None
    if raw is None and isinstance(x, dict) and "runtime_seconds" in x:
        raw = {"runtime_seconds": x["runtime_seconds"]}
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


def diagnostics(all=False):
    """Report wrapper runtime information, mirroring ``KODAMA.diagnostics``."""
    import kodama._core as _core_module

    extension = Path(_core_module.__file__)
    linker = "otool" if platform.system() == "Darwin" else "ldd"
    args = [linker, "-L", str(extension)] if linker == "otool" else [linker, str(extension)]
    linked = []
    try:
        linked = subprocess.run(args, check=False, capture_output=True, text=True).stdout.splitlines()
    except OSError:
        linked = []
    if not all:
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
    n_cores=4,
    n_iterations=10,
    random_walk_steps=4,
    gpu_device=0,
):
    """Cluster a graph or embedding, mirroring ``KODAMA.clustering``."""
    weight = _choice(weight, "weight", _GRAPH_WEIGHTS)
    metric = _choice(metric, "metric", _METRICS)
    graph_backend = _choice(graph_backend, "graph_backend", _BACKENDS)
    supplied_graph = _extract_graph(x)
    if supplied_graph is not None:
        return _graph_clustering(
            supplied_graph["indices"],
            supplied_graph["distances"],
            weight=weight,
            n_threads=int(n_cores),
            n_iterations=int(n_iterations),
            random_walk_steps=int(random_walk_steps),
            n_clusters=int(n_clusters),
        )
    return _embedding_clustering(
        _matrix32(x),
        graph_backend=graph_backend,
        weight=weight,
        metric=metric,
        k=int(k),
        n_threads=int(n_cores),
        n_iterations=int(n_iterations),
        random_walk_steps=int(random_walk_steps),
        n_clusters=int(n_clusters),
        gpu_device=int(gpu_device),
    )


KODAMA = SimpleNamespace(
    matrix=matrix,
    matrix_graph=matrix_graph,
    visualization=visualization,
    graph=graph,
    makeSNNGraph=graph,
    clustering=clustering,
    timing=timing,
    diagnostics=diagnostics,
    pca=kodama_pca,
)

knncv = KNNCV
plsldacv = PLSLDACV
core_knn = CoreKNN
core_plslda = CorePLSLDA
pca = kodama_pca
kodama_matrix = matrix
kodama_matrix_graph = matrix_graph
kodama_visualization = visualization
kodama_graph = graph
kodama_clustering = clustering
kodama_timing = timing
kodama_diagnostics = diagnostics

KODAMA_matrix = matrix
KODAMA_matrix_graph = matrix_graph
KODAMA_visualization = visualization
KODAMA_graph = graph
KODAMA_clustering = clustering
KODAMA_timing = timing
KODAMA_diagnostics = diagnostics
makeSNNGraph = graph
KODAMA_makeSNNGraph = graph
KODAMA_pca = kodama_pca

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
    "KODAMA_makeSNNGraph",
    "KodamaEmbedding",
    "KodamaMatrixResult",
    "makeSNNGraph",
    "clustering",
    "core_knn",
    "core_plslda",
    "diagnostics",
    "graph",
    "knncv",
    "kodama_clustering",
    "kodama_diagnostics",
    "kodama_graph",
    "kodama_matrix",
    "kodama_matrix_graph",
    "kodama_pca",
    "kodama_timing",
    "kodama_visualization",
    "matrix",
    "matrix_graph",
    "pca",
    "plsldacv",
    "timing",
    "visualization",
]
