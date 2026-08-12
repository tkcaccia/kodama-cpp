# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import os
import platform
import subprocess
import warnings
import inspect as _inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ._core import (
    core_knn as _core_knn,
    core_plslda as _core_plslda,
    embedding_clustering as _embedding_clustering,
    graph as _graph,
    graph_materialize as _graph_materialize,
    graph_clustering as _graph_clustering,
    knncv as _knncv,
    matrix as _matrix,
    matrix_graph as _matrix_graph,
    matrix_graph_handle as _matrix_graph_handle,
    normalization as _normalization,
    opentsne as _opentsne,
    passing_message as _passing_message,
    pca as _pca,
    pls as _pls,
    rsvd as _rsvd,
    scaling as _scaling,
    spatial_features as _spatial_features,
    plsldacv as _plsldacv,
    umap as _umap,
    visual_init as _visual_init,
)

_BACKENDS = ("cpu", "cuda", "metal")
_VISUALIZATION_BACKENDS = ("cpu", "cuda", "metal")
_METRICS = ("euclidean", "cosine", "inner_product")
_CLASSIFIERS = ("knn", "pls_lda")
_SPATIAL_CONSTRAINT_MODES = ("kmeans", "graph", "auto")
_SPATIAL_MODES = ("standard", "population")
_GRAPH_FEATURE_MODES = ("laplacian_self_tuning",)
_GRAPH_WEIGHTS = ("distance", "snn", "adaptive", "binary")
_EVOLUTION_POLICIES = (
    "full",
    "no_prediction_guidance",
    "fixed_proposal_budget",
    "no_transition_proposal",
    "greedy_acceptance",
    "raw_cv_score",
    "no_pls_transition_coarsening",
    "no_pls_fragmentation_penalty",
)


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


def _sample_ids(samples, n_samples):
    if samples is None:
        return None
    values = np.asarray(samples)
    if values.ndim != 1 or values.shape[0] != n_samples:
        raise ValueError("samples must have one value per data row")
    if any(value is None or value != value for value in values.tolist()):
        raise ValueError("samples must not contain missing values")
    _, encoded = np.unique(values, return_inverse=True)
    return (encoded + 1).astype(np.int32)


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
    """Cross-validated KNN classification, mirroring ``KODAMA::KNNCV``."""
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
    """Cross-validated SIMPLS plus LDA, mirroring ``KODAMA::PLSLDACV``."""
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
    """KODAMA label evolution with KNN, mirroring ``KODAMA::CoreKNN``."""
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
    """KODAMA label evolution with PLS-LDA, mirroring ``KODAMA::CorePLSLDA``."""
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
    """Backend-native float32 PCA, mirroring ``KODAMA::kodama_pca``."""
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


def rsvd(
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
    """Randomized truncated SVD using the standalone float32 core."""
    backend = _choice(backend, "backend", _BACKENDS)
    return _rsvd(
        _matrix32(data), ncomp=int(ncomp), center=bool(center),
        scale=bool(scale), backend=backend, seed=int(seed),
        n_threads=int(n_cores), gpu_device=int(gpu_device),
        oversample=-1 if oversample is None else int(oversample),
        power=-1 if power is None else int(power),
    )


def pls(
    data,
    response,
    ncomp=2,
    center=True,
    scale=True,
    backend="cpu",
    n_cores=1,
    gpu_device=0,
):
    """Fit a general float32 SIMPLS model."""
    backend = _choice(backend, "backend", _BACKENDS)
    return _pls(
        _matrix32(data), _matrix32(response, "response"),
        ncomp=int(ncomp), center=bool(center), scale=bool(scale),
        backend=backend, n_threads=int(n_cores), gpu_device=int(gpu_device),
    )


PCA = kodama_pca


def normalization(
    Xtrain,
    Xtest=None,
    method="pqn",
    ref=None,
    backend="cpu",
    n_cores=1,
    gpu_device=0,
):
    """Float32 KODAMA sample normalization."""
    method = _choice(method, "method", ("pqn", "sum", "median", "sqrt", "none"))
    backend = _choice(backend, "backend", _BACKENDS)
    return _normalization(
        _matrix32(Xtrain, "Xtrain"),
        test=None if Xtest is None else _matrix32(Xtest, "Xtest"),
        method=method,
        reference=None if ref is None else np.asarray(ref, dtype=np.float32),
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def scaling(
    Xtrain,
    Xtest=None,
    method="autoscaling",
    backend="cpu",
    n_cores=1,
    gpu_device=0,
):
    """Float32 KODAMA variable scaling using training statistics."""
    method = _choice(
        method,
        "method",
        ("none", "centering", "autoscaling", "rangescaling", "paretoscaling"),
    )
    backend = _choice(backend, "backend", _BACKENDS)
    return _scaling(
        _matrix32(Xtrain, "Xtrain"),
        test=None if Xtest is None else _matrix32(Xtest, "Xtest"),
        method=method,
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def spatial_feature_selection(
    data,
    spatial,
    samples=None,
    n_cores=4,
    require_nonzero_each_sample=True,
):
    """Screen spatially autocorrelated features on the multicore CPU backend."""
    data = _matrix32(data)
    spatial = _matrix32(spatial, "spatial")
    if spatial.shape[0] != data.shape[0]:
        raise ValueError("data and spatial must have the same rows")
    result = _spatial_features(
        data,
        spatial,
        samples=_sample_ids(samples, data.shape[0]),
        n_threads=int(n_cores),
        require_nonzero_each_sample=bool(require_nonzero_each_sample),
    )
    result["features"] = result["ranking"].copy()
    return result


def passing_message(
    data,
    spatial,
    samples=None,
    number_knn=15,
    backend="cpu",
    n_cores=4,
    gpu_device=0,
):
    """Aggregate local expression messages within each sample or slide."""
    data = _matrix32(data)
    spatial = _matrix32(spatial, "spatial")
    if spatial.shape[0] != data.shape[0]:
        raise ValueError("data and spatial must have the same rows")
    backend = _choice(backend, "backend", _BACKENDS)
    return _passing_message(
        data,
        spatial,
        samples=_sample_ids(samples, data.shape[0]),
        number_knn=int(number_knn),
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
    )


def _dense_object_matrix(value, name="data"):
    """Convert NumPy or scipy-like matrices without importing scipy."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return _matrix32(value, name)


def _anndata_table(value, table_key=None):
    """Return an AnnData-like table from AnnData or SpatialData."""
    if hasattr(value, "X") and hasattr(value, "obs") and hasattr(value, "obsm"):
        return value
    tables = getattr(value, "tables", None)
    if tables is None:
        raise TypeError("object must be AnnData-like or SpatialData-like")
    keys = list(tables.keys())
    if table_key is None:
        if len(keys) != 1:
            raise ValueError("table_key is required when SpatialData has multiple tables")
        table_key = keys[0]
    if table_key not in tables:
        raise KeyError(f"SpatialData table {table_key!r} was not found")
    return tables[table_key]


def _anndata_values(adata, layer=None):
    value = adata.X if layer is None else adata.layers[layer]
    return _dense_object_matrix(value)


def _annotation_vector(frame, key, n_samples, name):
    if key is None:
        return None
    try:
        values = np.asarray(frame[key])
    except Exception as error:
        raise KeyError(f"{name} key {key!r} was not found") from error
    if values.ndim != 1 or values.shape[0] != n_samples:
        raise ValueError(f"{name} must contain one value per observation")
    return values


def _sparse_knn_graph(matrix, k=100, similarity=False):
    """Convert a scipy-compatible sparse matrix to fixed-width KNN arrays."""
    if not all(hasattr(matrix, field) for field in ("indptr", "indices", "data")):
        raise TypeError("Squidpy graph must be a CSR-compatible sparse matrix")
    csr = matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
    n_samples = int(csr.shape[0])
    width = min(int(k), max(1, n_samples - 1))
    indices = np.empty((n_samples, width), dtype=np.int32)
    distances = np.empty((n_samples, width), dtype=np.float32)
    for row in range(n_samples):
        begin, end = int(csr.indptr[row]), int(csr.indptr[row + 1])
        candidates = np.asarray(csr.indices[begin:end], dtype=np.int32)
        values = np.asarray(csr.data[begin:end], dtype=np.float32)
        keep = candidates != row
        candidates, values = candidates[keep], values[keep]
        if similarity:
            values = 1.0 - np.clip(values, 0.0, 1.0)
        order = np.argsort(values, kind="stable")
        candidates, values = candidates[order], values[order]
        if candidates.size == 0:
            candidates = np.array([row], dtype=np.int32)
            values = np.array([0.0], dtype=np.float32)
        take = min(width, candidates.size)
        indices[row, :take] = candidates[:take]
        distances[row, :take] = values[:take]
        if take < width:
            indices[row, take:] = candidates[take - 1]
            distances[row, take:] = values[take - 1]
    return {"indices": indices, "distances": distances, "neighbors": width}


def graph_from_anndata(
    obj,
    *,
    table_key=None,
    distances_key=None,
    connectivities_key=None,
    k=100,
):
    """Read a Scanpy/Squidpy graph from ``AnnData.obsp``."""
    adata = _anndata_table(obj, table_key)
    candidates = []
    if distances_key is not None:
        candidates.append((distances_key, False))
    else:
        candidates.extend((("spatial_distances", False), ("distances", False)))
    if connectivities_key is not None:
        candidates.append((connectivities_key, True))
    else:
        candidates.extend((("spatial_connectivities", True), ("connectivities", True)))
    for key, similarity in candidates:
        try:
            value = adata.obsp[key]
        except Exception:
            continue
        out = _sparse_knn_graph(value, k=k, similarity=similarity)
        out.update({"source": "anndata_obsp", "key": key, "storage": "matrix"})
        return out
    raise KeyError("no Scanpy/Squidpy distance or connectivity graph was found in obsp")


def RunKODAMAgraph(
    obj,
    *,
    table_key=None,
    layer=None,
    spatial_key="spatial",
    sample_key=None,
    result_key="kodama_graph",
    copy=False,
    **kwargs,
):
    """Build and store a KODAMA graph on AnnData or SpatialData."""
    target = obj.copy() if copy and hasattr(obj, "copy") else obj
    adata = _anndata_table(target, table_key)
    data = _anndata_values(adata, layer)
    spatial = None if spatial_key is None else _matrix32(adata.obsm[spatial_key], "spatial")
    samples = _annotation_vector(adata.obs, sample_key, data.shape[0], "sample")
    result = graph(data, spatial=spatial, samples=samples, **kwargs)
    adata.uns[result_key] = result
    return target if copy else result


def RunKODAMAmatrix(
    obj,
    *,
    table_key=None,
    layer=None,
    spatial_key="spatial",
    sample_key=None,
    graph_key="kodama_graph",
    result_key="kodama",
    labels_key="kodama_labels",
    use_existing_graph=True,
    copy=False,
    **kwargs,
):
    """Run KODAMA and store best labels on AnnData or SpatialData."""
    target = obj.copy() if copy and hasattr(obj, "copy") else obj
    adata = _anndata_table(target, table_key)
    data = _anndata_values(adata, layer)
    spatial = None if spatial_key is None else _matrix32(adata.obsm[spatial_key], "spatial")
    samples = _annotation_vector(adata.obs, sample_key, data.shape[0], "sample")
    prepared = adata.uns.get(graph_key) if use_existing_graph else None
    kwargs.setdefault("return_graph", True)
    result = matrix(data=data, graph=prepared, spatial=spatial, samples=samples, **kwargs)
    adata.uns[result_key] = result
    adata.obs[labels_key] = np.asarray(result.best_labels).astype(str)
    return target if copy else result


def RunKODAMAvisualization(
    obj,
    *,
    table_key=None,
    layer=None,
    result_key="kodama",
    embedding_key="X_kodama_umap",
    method="UMAP",
    copy=False,
    **kwargs,
):
    """Embed a stored KODAMA result and write it to ``AnnData.obsm``."""
    target = obj.copy() if copy and hasattr(obj, "copy") else obj
    adata = _anndata_table(target, table_key)
    if result_key not in adata.uns:
        raise KeyError(f"KODAMA result {result_key!r} was not found in uns")
    embedding = visualization(
        adata.uns[result_key], method=method,
        raw_data=_anndata_values(adata, layer), **kwargs
    )
    adata.obsm[embedding_key] = np.asarray(embedding, dtype=np.float32)
    return target if copy else embedding


def RunSpatialFeatureSelection(
    obj,
    *,
    table_key=None,
    layer=None,
    spatial_key="spatial",
    sample_key=None,
    result_key="kodama_spatial_features",
    copy=False,
    **kwargs,
):
    """Run spatial feature selection and annotate ``AnnData.var``."""
    target = obj.copy() if copy and hasattr(obj, "copy") else obj
    adata = _anndata_table(target, table_key)
    data = _anndata_values(adata, layer)
    samples = _annotation_vector(adata.obs, sample_key, data.shape[0], "sample")
    result = spatial_feature_selection(
        data, _matrix32(adata.obsm[spatial_key], "spatial"), samples=samples, **kwargs
    )
    adata.uns[result_key] = result
    for key in ("score", "p_value", "adjusted_p_value"):
        adata.var[f"{result_key}_{key}"] = np.asarray(result[key])
    return target if copy else result


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
    """Float32 embedding with observable execution and graph provenance."""

    def __new__(
        cls,
        values,
        initialization,
        initialization_backend,
        *,
        backend=None,
        optimizer=None,
        graph_edges=0,
        graph_max_weight=0.0,
        runtime_seconds=np.nan,
    ):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.initialization = initialization
        obj.initialization_backend = initialization_backend
        obj.backend = backend
        obj.optimizer = optimizer
        obj.graph_edges = int(graph_edges)
        obj.graph_max_weight = float(graph_max_weight)
        obj.runtime_seconds = float(runtime_seconds)
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self.initialization = getattr(source, "initialization", None)
        self.initialization_backend = getattr(
            source, "initialization_backend", None
        )
        self.backend = getattr(source, "backend", None)
        self.optimizer = getattr(source, "optimizer", None)
        self.graph_edges = getattr(source, "graph_edges", 0)
        self.graph_max_weight = getattr(source, "graph_max_weight", 0.0)
        self.runtime_seconds = getattr(source, "runtime_seconds", np.nan)


def _embedding_from_native(result, initialization, initialization_backend):
    """Convert the extension result while accepting pre-diagnostic builds."""
    if isinstance(result, dict):
        return KodamaEmbedding(
            result["embedding"],
            initialization,
            initialization_backend,
            backend=result.get("backend"),
            optimizer=result.get("optimizer"),
            graph_edges=result.get("graph_edges", 0),
            graph_max_weight=result.get("graph_max_weight", 0.0),
            runtime_seconds=result.get("runtime_seconds", np.nan),
        )
    return KodamaEmbedding(
        result,
        initialization,
        initialization_backend,
    )


def _default_splitting(n_samples):
    return 100 if n_samples < 40000 else 300


def matrix(
    data=None,
    graph=None,
    spatial=None,
    samples=None,
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
    spatial_mode="standard",
    metric="euclidean",
    classifier="knn",
    backend=None,
    seed=1234,
    folds=5,
    visual_init=True,
    progress=True,
    apply_kodama_dissimilarity=True,
    return_graph=False,
    _evolution_policy="full",
):
    """Run KODAMA matrix optimization, mirroring ``KODAMA::kodama_matrix``.

    The native call builds one full-data graph before all ``M`` runs. With
    ``visual_init=True`` it also performs one float32 PCA and stores both UMAP
    and openTSNE starts for :func:`visualization`.
    """
    supplied_graph = _extract_graph(graph)
    if graph is not None:
        if supplied_graph is None:
            raise ValueError(
                "graph must be a KODAMA.graph result or a dictionary "
                "with indices and distances"
            )
        if _extract_graph(data) is not None:
            raise ValueError(
                "data must be the raw array; pass graph inputs through graph"
            )
        raw = data
        n_samples = (
            int(supplied_graph["samples"])
            if "handle" in supplied_graph
            else np.asarray(supplied_graph["indices"]).shape[0]
        )
        if raw is not None and np.asarray(raw).shape[0] != n_samples:
            raise ValueError(
                "data and graph must contain the same number of samples"
            )
        if ncomp is None:
            ncomp = 50 if raw is None else min(50, np.asarray(raw).shape[1])
        if splitting is None:
            splitting = _default_splitting(n_samples)
        if graph_neighbors is None:
            graph_neighbors = np.asarray(supplied_graph["indices"]).shape[1]
        if backend is None:
            backend = graph.get("backend", "cpu")
        return matrix_graph(
            graph,
            data=raw,
            spatial=spatial,
            samples=samples,
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
            spatial_mode=spatial_mode,
            classifier=classifier,
            backend=backend,
            seed=seed,
            folds=folds,
            visual_init=visual_init,
            progress=progress,
            apply_kodama_dissimilarity=apply_kodama_dissimilarity,
            return_graph=return_graph,
            _evolution_policy=_evolution_policy,
        )

    if data is None:
        raise ValueError("data or graph is required")
    if _extract_graph(data) is not None:
        raise ValueError(
            "data must be the raw array; pass graph inputs through graph"
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
    _evolution_policy = _choice(
        _evolution_policy, "_evolution_policy", _EVOLUTION_POLICIES
    )
    spatial_constraint_code = _spatial_constraint_code(spatial_constraint_mode)
    spatial_mode = _choice(spatial_mode, "spatial_mode", _SPATIAL_MODES)
    spatial_array = None if spatial is None else _matrix32(spatial, "spatial")
    sample_ids = _sample_ids(samples, x.shape[0])
    result = _matrix(
        x,
        spatial=spatial_array,
        samples=sample_ids,
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
        spatial_coordinate_mode=1 if spatial_mode == "population" else 0,
        metric=metric,
        classifier=classifier,
        backend=backend,
        seed=int(seed),
        progress=bool(progress),
        apply_kodama_dissimilarity=bool(apply_kodama_dissimilarity),
        compute_visual_init=bool(visual_init),
        return_graph=bool(return_graph),
        folds=int(folds),
        evolution_policy=str(_evolution_policy),
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
        "samples": 0 if sample_ids is None else int(np.unique(sample_ids).size),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": spatial_constraint_mode,
        "spatial_mode": spatial_mode,
        "metric": metric,
        "classifier": classifier,
        "backend": backend,
        "seed": int(seed),
        "folds": int(folds),
        "evolution_policy": str(_evolution_policy),
        "visual_init": bool(visual_init),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
        "return_graph": bool(return_graph),
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
    samples=None,
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
    spatial_mode="standard",
    classifier="knn",
    backend=None,
    graph_feature_mode="laplacian_self_tuning",
    graph_feature_components=0,
    graph_feature_steps=3,
    seed=1234,
    folds=5,
    visual_init=True,
    progress=True,
    apply_kodama_dissimilarity=True,
    return_graph=False,
    _evolution_policy="full",
):
    """Run KODAMA from a KNN matrix, mirroring ``KODAMA::kodama_matrix_graph``."""
    graph_object = indices if isinstance(indices, dict) else None
    supplied_graph = _extract_graph(indices)
    graph_handle = supplied_graph.get("handle") if isinstance(supplied_graph, dict) else None
    if supplied_graph is not None:
        distances = supplied_graph.get("distances", distances)
        indices = supplied_graph.get("indices")
    if graph_handle is None and (indices is None or distances is None):
        raise ValueError("indices and distances are required")
    idx = None if graph_handle is not None else np.asarray(indices, dtype=np.int32)
    dst = None if graph_handle is not None else np.asarray(distances, dtype=np.float32)
    if graph_handle is None and (idx.ndim != 2 or dst.ndim != 2 or idx.shape != dst.shape):
        raise ValueError("indices and distances must be matrices with the same shape")
    if graph_neighbors is None:
        graph_neighbors = int(supplied_graph.get("neighbors")) if graph_handle is not None else idx.shape[1]
    if ncomp is None:
        ncomp = 50 if data is None else min(50, np.asarray(data).shape[1])
    if splitting is None:
        splitting = _default_splitting(int(supplied_graph["samples"]) if graph_handle is not None else idx.shape[0])
    classifier = _choice(classifier, "classifier", _CLASSIFIERS)
    if backend is None:
        backend = graph_object.get("backend", "cpu") if isinstance(graph_object, dict) else "cpu"
    backend = _choice(backend, "backend", _BACKENDS)
    graph_feature_mode = _choice(
        graph_feature_mode, "graph_feature_mode", _GRAPH_FEATURE_MODES
    )
    _evolution_policy = _choice(
        _evolution_policy, "_evolution_policy", _EVOLUTION_POLICIES
    )
    spatial_constraint_code = _spatial_constraint_code(spatial_constraint_mode)
    spatial_mode = _choice(spatial_mode, "spatial_mode", _SPATIAL_MODES)
    sample_ids = _sample_ids(
        samples,
        int(supplied_graph["samples"]) if graph_handle is not None else idx.shape[0],
    )
    native = _matrix_graph_handle if graph_handle is not None else _matrix_graph
    positional = (graph_handle,) if graph_handle is not None else (idx, dst)
    result = native(
        *positional,
        data=None if data is None else _matrix32(data),
        spatial=None if spatial is None else _matrix32(spatial, "spatial"),
        samples=sample_ids,
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
        spatial_coordinate_mode=1 if spatial_mode == "population" else 0,
        classifier=classifier,
        backend=backend,
        graph_feature_mode=graph_feature_mode,
        graph_feature_components=int(graph_feature_components),
        graph_feature_steps=int(graph_feature_steps),
        seed=int(seed),
        progress=bool(progress),
        apply_kodama_dissimilarity=bool(apply_kodama_dissimilarity),
        return_graph=bool(return_graph),
        folds=int(folds),
        evolution_policy=str(_evolution_policy),
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
        "samples": 0 if sample_ids is None else int(np.unique(sample_ids).size),
        "spatial_graph_mix": bool(spatial_graph_mix),
        "spatial_constraint_mode": spatial_constraint_mode,
        "spatial_mode": spatial_mode,
        "classifier": classifier,
        "backend": backend,
        "graph_feature_mode": graph_feature_mode,
        "graph_feature_components": int(graph_feature_components),
        "graph_feature_steps": int(graph_feature_steps),
        "graph_uses_data_geometry": data is not None,
        "seed": int(seed),
        "folds": int(folds),
        "evolution_policy": str(_evolution_policy),
        "visual_init": bool(visual_init),
        "apply_kodama_dissimilarity": bool(apply_kodama_dissimilarity),
        "return_graph": bool(return_graph),
    }
    result["class_counts"] = result.class_counts
    result["best_run"] = result.best_run
    result["best_labels"] = result.best_labels
    return result


def graph(
    data,
    spatial=None,
    samples=None,
    k=100,
    metric="euclidean",
    backend="cpu",
    n_cores=4,
    gpu_device=0,
    seed=1234,
    storage="handle",
):
    """Build a reusable KODAMA graph and backend-specific PCA starts."""
    metric = _choice(metric, "metric", _METRICS)
    backend = _choice(backend, "backend", _BACKENDS)
    x = _matrix32(data)
    spatial_array = None if spatial is None else _matrix32(spatial, "spatial")
    sample_ids = _sample_ids(samples, x.shape[0])
    result = _graph(
        x,
        spatial=spatial_array,
        samples=sample_ids,
        k=int(k),
        metric=metric,
        backend=backend,
        n_threads=int(n_cores),
        gpu_device=int(gpu_device),
        seed=int(seed),
        storage=storage,
    )
    result["parameters"] = {
        "k": int(k),
        "spatial": spatial is not None,
        "samples": 0 if sample_ids is None else int(np.unique(sample_ids).size),
        "metric": metric,
        "backend": backend,
        "n_cores": int(n_cores),
        "gpu_device": int(gpu_device),
        "seed": int(seed),
    }
    return result


def graph_materialize(value):
    """Explicitly copy a handle-backed graph into NumPy index/distance arrays."""
    graph_object = _extract_graph(value)
    if graph_object is None:
        raise ValueError("value is not a KODAMA graph")
    if "handle" not in graph_object:
        return graph_object
    out = _graph_materialize(graph_object["handle"])
    out.update({k: v for k, v in graph_object.items() if k not in ("handle", "indices", "distances")})
    out["storage"] = "matrix"
    return out


def _extract_graph(value):
    if isinstance(value, dict) and "knn" in value:
        return value["knn"]
    if isinstance(value, dict) and "handle" in value:
        return value
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
    method_l = method.lower()
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
    if "handle" in g:
        g = graph_materialize(g)

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
        return _embedding_from_native(
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
        return _embedding_from_native(
            values,
            initialization or "random",
            initialization_backend or backend,
        )
    raise ValueError(f"Unsupported visualization method: {method}")


def umap(x, **kwargs):
    """Run the native UMAP implementation on data or a prepared KNN graph."""
    return visualization(x, method="UMAP", **kwargs)


def opentsne(x, **kwargs):
    """Run the native openTSNE implementation on data or a prepared KNN graph."""
    return visualization(x, method="opentsne", **kwargs)


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
        if "handle" in supplied_graph:
            supplied_graph = graph_materialize(supplied_graph)
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


def _hide_experimental_signature(function):
    signature = _inspect.signature(function)
    function.__signature__ = signature.replace(parameters=[
        parameter for parameter in signature.parameters.values()
        if parameter.name != "_evolution_policy"
    ])


_hide_experimental_signature(matrix)
_hide_experimental_signature(matrix_graph)


KODAMA = SimpleNamespace(
    matrix=matrix,
    matrix_graph=matrix_graph,
    visualization=visualization,
    umap=umap,
    opentsne=opentsne,
    graph=graph,
    makeSNNGraph=graph,
    clustering=clustering,
    timing=timing,
    diagnostics=diagnostics,
    pca=kodama_pca,
    pls=pls,
    rsvd=rsvd,
    normalization=normalization,
    scaling=scaling,
    spatial_features=spatial_feature_selection,
    passing_message=passing_message,
)

knncv = KNNCV
plsldacv = PLSLDACV
core_knn = CoreKNN
core_plslda = CorePLSLDA
pca = kodama_pca
spatial_features = spatial_feature_selection
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
KODAMA_spatial_features = spatial_feature_selection
KODAMA_passing_message = passing_message

__all__ = [
    "KNNCV",
    "PLSLDACV",
    "CoreKNN",
    "CorePLSLDA",
    "PCA",
    "pls",
    "rsvd",
    "normalization",
    "scaling",
    "passing_message",
    "spatial_feature_selection",
    "spatial_features",
    "KODAMA",
    "KODAMA_matrix",
    "KODAMA_matrix_graph",
    "KODAMA_visualization",
    "KODAMA_graph",
    "KODAMA_clustering",
    "KODAMA_timing",
    "KODAMA_diagnostics",
    "KODAMA_pca",
    "KODAMA_spatial_features",
    "KODAMA_passing_message",
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
    "umap",
    "opentsne",
    "graph_from_anndata",
    "RunKODAMAgraph",
    "RunKODAMAmatrix",
    "RunKODAMAvisualization",
    "RunSpatialFeatureSelection",
]
