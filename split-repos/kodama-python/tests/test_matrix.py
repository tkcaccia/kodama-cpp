# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import inspect

import numpy as np
import pytest

import kodama


def test_native_preprocessing_cpu():
    train = np.array([[1, 2, 3], [2, 4, 8], [4, 1, 5], [3, 6, 9]], dtype=np.float32)
    test = np.array([[2, 3, 4], [5, 2, 1]], dtype=np.float32)
    for method in ("pqn", "sum", "median", "sqrt", "none"):
        result = kodama.normalization(train, test, method=method, n_cores=4)
        np.testing.assert_allclose(
            result["newXtrain"] * result["coeXtrain"][:, None], train, rtol=2e-6, atol=2e-6
        )
        assert result["backend"] == "cpu"
        assert result["precision"] == "float32"
    for method in ("none", "centering", "autoscaling", "rangescaling", "paretoscaling"):
        result = kodama.scaling(train, test, method=method, n_cores=4)
        np.testing.assert_allclose(
            result["newXtrain"] * result["scale"] + result["center"],
            train,
            rtol=2e-6,
            atol=2e-6,
        )


def test_matrix_knn_and_pls_lda_cpu(monkeypatch):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(90, 6)).astype(np.float32)
    spatial = rng.normal(size=(90, 2)).astype(np.float32)

    knn = kodama.matrix(
        x,
        spatial=spatial,
        M=1,
        Tcycle=1,
        landmarks=60,
        classifier="knn",
        backend="cpu",
        progress=False,
    )
    pls = kodama.matrix(
        x,
        spatial=spatial,
        M=1,
        Tcycle=1,
        ncomp=3,
        landmarks=60,
        classifier="pls_lda",
        backend="cpu",
        progress=False,
    )

    assert knn["res"].shape == (1, 90)
    assert pls["res"].shape == (1, 90)
    assert knn["analysis_storage"] == "float32"
    assert pls["analysis_storage"] == "float32"
    assert isinstance(knn, kodama.KodamaMatrixResult)
    assert knn["visual_init"]["backend"] == "cpu"
    assert knn["visual_init"]["precision"] == "float32"
    assert knn["visual_init"]["umap"].shape == (90, 2)
    assert knn["visual_init"]["opentsne"].shape == (90, 2)
    assert knn["graph_builds"] == 1
    assert knn["knn_is_kodama_corrected"] is True
    assert "base_knn" not in knn
    assert knn["graph_storage_bytes"] >= (
        knn["knn"]["indices"].size * 4 + knn["knn"]["distances"].size * 4
    )
    assert knn["timing"]["visual_init_seconds"] >= 0
    assert knn.best_labels.shape == (90,)
    assert knn["best_labels"].shape == (90,)
    assert knn.class_counts.shape == (1,)
    assert knn.best_run == knn["best_run"]
    assert knn["parameters"]["classifier"] == "knn"
    assert knn["parameters"]["spatial_resolution"] == 0.4
    assert knn["parameters"]["spatial_constraint_mode"] == "kmeans"
    assert any(row["step"] == "runtime_seconds" for row in kodama.timing(knn))
    umap_layout = kodama.visualization(
        knn,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
    )
    tsne_layout = kodama.visualization(
        knn,
        method="opentsne",
        k=5,
        perplexity=1,
        early_exaggeration_iter=1,
        n_iter=2,
        backend="cpu",
    )
    assert umap_layout.shape == (90, 2)
    assert tsne_layout.shape == (90, 2)
    assert umap_layout.initialization == "raw_pca"
    assert umap_layout.initialization_backend == "cpu"
    assert tsne_layout.initialization == "raw_pca"
    assert tsne_layout.initialization_backend == "cpu"
    assert np.all(np.isfinite(umap_layout))
    assert np.all(np.isfinite(tsne_layout))

    def fail_recomputed_initialization(*args, **kwargs):
        raise AssertionError("stored initialization was not reused")

    monkeypatch.setattr(kodama, "_visual_initialization", fail_recomputed_initialization)
    umap_with_raw = kodama.visualization(
        knn,
        method="UMAP",
        raw_data=x,
        k=5,
        n_epochs=1,
        backend="cpu",
    )
    assert umap_with_raw.initialization == "raw_pca"


def test_public_api_cpu():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 5)).astype(np.float32)
    labels = np.resize(np.arange(1, 4, dtype=np.int32), x.shape[0])

    knncv = kodama.KNNCV(x, labels, folds=3, k=3, backend="cpu")
    pls = kodama.PLSLDACV(x, labels, folds=3, ncomp=2, backend="cpu")
    core_knn = kodama.CoreKNN(x, labels, cycles=1, folds=3, k=3, backend="cpu")
    core_pls = kodama.CorePLSLDA(x, labels, cycles=1, folds=3, ncomp=2, backend="cpu")
    pca = kodama.PCA(x, ncomp=3, backend="cpu", seed=4)
    graph = kodama.graph(x, k=5, backend="cpu")
    emb_default = kodama.visualization(
        graph,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
    )
    emb_fuzzy = kodama.visualization(
        graph,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
        graph_mode="fuzzy",
    )
    emb_binary = kodama.visualization(
        graph,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
        graph_mode="binary",
    )
    emb_raw = kodama.visualization(
        x,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
    )
    tsne_raw = kodama.visualization(
        x,
        method="opentsne",
        k=5,
        perplexity=1,
        early_exaggeration_iter=1,
        n_iter=2,
        backend="cpu",
    )
    clu = kodama.clustering(graph, n_iterations=2, random_walk_steps=2)

    assert knncv["predicted"].shape == (60,)
    assert pls["predicted"].shape == (60,)
    assert core_knn["clbest"].shape == (60,)
    assert core_pls["clbest"].shape == (60,)
    assert pca["scores"].shape == (60, 3)
    assert pca["loadings"].shape == (5, 3)
    assert pca["scores"].dtype == np.float32
    assert pca["precision"] == "float32"
    assert np.all(np.diff(pca["singular_values"]) <= 1e-5)
    assert graph["indices"].shape == (60, 5)
    assert "data" not in graph
    assert graph["visual_init"]["umap"].shape == (60, 2)
    assert graph["visual_init"]["opentsne"].shape == (60, 2)
    assert emb_default.shape == (60, 2)
    assert emb_fuzzy.shape == (60, 2)
    assert emb_binary.shape == (60, 2)
    assert emb_raw.shape == (60, 2)
    assert tsne_raw.shape == (60, 2)
    assert emb_default.initialization == "raw_pca"
    assert emb_default.initialization_backend == "cpu"
    assert emb_raw.initialization == "raw_pca"
    assert tsne_raw.initialization == "raw_pca"
    np.testing.assert_array_equal(emb_default, emb_fuzzy)
    assert np.all(np.isfinite(emb_default))
    assert np.all(np.isfinite(emb_fuzzy))
    assert np.all(np.isfinite(emb_binary))
    assert np.all(np.isfinite(emb_raw))
    assert np.all(np.isfinite(tsne_raw))
    assert clu["membership"].shape == (60,)


def test_matrix_graph_cpu():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 7)).astype(np.float32)
    g = kodama.graph(x, k=15, backend="cpu", n_cores=1, seed=3)

    knn = kodama.matrix_graph(
        g,
        M=1,
        Tcycle=1,
        landmarks=50,
        splitting=5,
        n_cores=1,
        graph_neighbors=15,
        knn_k=5,
        classifier="knn",
        backend="cpu",
        progress=False,
    )
    pls = kodama.matrix_graph(
        g,
        M=1,
        Tcycle=1,
        ncomp=3,
        landmarks=50,
        splitting=5,
        graph_neighbors=15,
        knn_k=5,
        classifier="pls_lda",
        backend="cpu",
        graph_feature_components=4,
        graph_feature_steps=2,
        progress=False,
    )

    assert knn["res"].shape == (1, 80)
    assert pls["res"].shape == (1, 80)
    assert knn["parameters"]["classifier"] == "knn"
    assert pls["parameters"]["graph_feature_mode"] == "laplacian_self_tuning"
    assert any(row["step"] == "graph_feature_seconds" for row in kodama.timing(pls))

    bare = {"indices": g["indices"], "distances": g["distances"]}
    common = dict(
        M=1,
        Tcycle=1,
        ncomp=3,
        landmarks=50,
        splitting=5,
        graph_neighbors=15,
        knn_k=5,
        classifier="knn",
        backend="cpu",
        seed=3,
        progress=False,
    )
    from_raw = kodama.matrix(data=x, **common)
    from_prepared = kodama.matrix(graph=g, **common)
    from_prepared_data = kodama.matrix(data=x, graph=g, **common)
    from_bare = kodama.matrix(graph=bare, **common)

    assert from_raw["graph_builds"] == 1
    assert from_prepared["graph_builds"] == 0
    assert from_prepared_data["graph_builds"] == 0
    assert from_bare["graph_builds"] == 0
    assert from_prepared["parameters"]["graph_uses_data_geometry"] is False
    assert from_prepared_data["parameters"]["graph_uses_data_geometry"] is True
    assert from_bare["parameters"]["graph_uses_data_geometry"] is False
    assert from_prepared_data["res"].shape == from_raw["res"].shape
    np.testing.assert_array_equal(
        from_prepared["visual_init"]["umap"], g["visual_init"]["umap"]
    )
    assert "visual_init" not in from_bare or from_bare["visual_init"] is None
    with pytest.raises(ValueError, match="pass graph inputs through graph"):
        kodama.matrix(data=g, **common)


def test_diagnostics():
    diag = kodama.diagnostics()
    assert "extension" in diag
    assert "CONDA_PREFIX" in diag["environment"]


def test_python_signatures_mirror_r_api():
    assert list(inspect.signature(kodama.KNNCV).parameters) == [
        "data",
        "labels",
        "constrain",
        "folds",
        "stratified",
        "seed",
        "k",
        "metric",
        "backend",
        "n_cores",
        "gpu_device",
    ]
    assert list(inspect.signature(kodama.PLSLDACV).parameters) == [
        "data",
        "labels",
        "constrain",
        "folds",
        "stratified",
        "seed",
        "ncomp",
        "center",
        "scale",
        "backend",
        "n_cores",
        "gpu_device",
    ]
    assert list(inspect.signature(kodama.CoreKNN).parameters) == [
        "data",
        "labels",
        "constrain",
        "fix",
        "cycles",
        "folds",
        "stratified",
        "seed",
        "k",
        "metric",
        "backend",
        "n_cores",
        "gpu_device",
    ]
    assert list(inspect.signature(kodama.CorePLSLDA).parameters) == [
        "data",
        "labels",
        "constrain",
        "fix",
        "cycles",
        "folds",
        "stratified",
        "seed",
        "ncomp",
        "backend",
        "n_cores",
        "gpu_device",
    ]

    matrix_signature = inspect.signature(kodama.matrix)
    assert list(matrix_signature.parameters) == [
        "data",
        "graph",
        "spatial",
        "W",
        "constrain",
        "fix",
        "M",
        "Tcycle",
        "ncomp",
        "landmarks",
        "splitting",
        "n_cores",
        "graph_neighbors",
        "knn_k",
        "spatial_resolution",
        "spatial_graph_mix",
        "spatial_constraint_mode",
        "metric",
        "classifier",
        "backend",
        "seed",
        "visual_init",
        "progress",
        "apply_kodama_dissimilarity",
    ]
    assert matrix_signature.parameters["spatial_resolution"].default == 0.4
    assert matrix_signature.parameters["spatial_constraint_mode"].default == "kmeans"
    assert "n_cores" in matrix_signature.parameters
    assert "n_threads" not in matrix_signature.parameters

    graph_signature = inspect.signature(kodama.matrix_graph)
    assert list(graph_signature.parameters) == [
        "indices",
        "distances",
        "data",
        "spatial",
        "W",
        "constrain",
        "fix",
        "M",
        "Tcycle",
        "ncomp",
        "landmarks",
        "splitting",
        "n_cores",
        "graph_neighbors",
        "knn_k",
        "spatial_resolution",
        "spatial_graph_mix",
        "spatial_constraint_mode",
        "classifier",
        "backend",
        "graph_feature_mode",
        "graph_feature_components",
        "graph_feature_steps",
        "seed",
        "visual_init",
        "progress",
        "apply_kodama_dissimilarity",
    ]
    assert graph_signature.parameters["spatial_resolution"].default == 0.4
    assert graph_signature.parameters["spatial_constraint_mode"].default == "kmeans"
    assert "n_cores" in graph_signature.parameters

    visualization_signature = inspect.signature(kodama.visualization)
    assert list(visualization_signature.parameters) == [
        "x",
        "method",
        "init",
        "raw_data",
        "initialize_from_raw",
        "k",
        "metric",
        "backend",
        "n_cores",
        "gpu_device",
        "n_epochs",
        "n_iter",
        "perplexity",
        "graph_mode",
        "seed",
        "kwargs",
    ]
    assert visualization_signature.parameters["seed"].default == 4
    assert visualization_signature.parameters["graph_mode"].default == "fuzzy"
    assert "n_cores" in visualization_signature.parameters
    assert list(inspect.signature(kodama.graph).parameters) == [
        "data",
        "k",
        "metric",
        "backend",
        "n_cores",
        "gpu_device",
        "seed",
    ]
    assert list(inspect.signature(kodama.kodama_pca).parameters) == [
        "data",
        "ncomp",
        "center",
        "scale",
        "backend",
        "n_cores",
        "gpu_device",
        "seed",
        "oversample",
        "power",
    ]
    assert list(inspect.signature(kodama.clustering).parameters) == [
        "x",
        "n_clusters",
        "weight",
        "k",
        "metric",
        "graph_backend",
        "n_cores",
        "n_iterations",
        "random_walk_steps",
        "gpu_device",
    ]
    assert list(inspect.signature(kodama.timing).parameters) == ["x"]
    assert list(inspect.signature(kodama.diagnostics).parameters) == ["all"]

    assert kodama.knncv is kodama.KNNCV
    assert kodama.plsldacv is kodama.PLSLDACV
    assert kodama.core_knn is kodama.CoreKNN
    assert kodama.core_plslda is kodama.CorePLSLDA
    assert kodama.KODAMA.matrix is kodama.matrix
    assert kodama.KODAMA.matrix_graph is kodama.matrix_graph
    assert kodama.KODAMA.visualization is kodama.visualization
    assert kodama.KODAMA.makeSNNGraph is kodama.graph
    assert kodama.KODAMA.pca is kodama.kodama_pca


def test_python_high_level_validation_matches_r_choices():
    x = np.zeros((8, 2), dtype=np.float32)
    labels = np.resize(np.arange(1, 3, dtype=np.int32), x.shape[0])

    with pytest.raises(ValueError, match="spatial_constraint_mode"):
        kodama.matrix(
            x,
            M=1,
            Tcycle=1,
            spatial_constraint_mode=0,
            progress=False,
        )
    with pytest.raises(ValueError, match="method"):
        kodama.visualization(x, method="umap")
    with pytest.raises(TypeError):
        kodama.KNNCV(x, labels, n_threads=1)
