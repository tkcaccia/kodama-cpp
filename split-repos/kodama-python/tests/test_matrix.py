# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import inspect
import platform

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
        return_graph=True,
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
    assert knn["n_cores"] == 4
    assert knn["gpu_auto_workers"] is False
    assert knn["gpu_scheduler_enabled"] is False
    assert knn["gpu_scheduler_lanes"] == 0
    assert knn["gpu_worker_memory_estimate_mb"] == 0.0
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


def test_reviewer_evolution_policies_are_reproducible_but_hidden():
    rng = np.random.default_rng(53)
    x = rng.normal(size=(150, 6)).astype(np.float32)
    prepared = kodama.graph(
        x, k=12, backend="cpu", n_cores=2, seed=19, storage="handle"
    )
    common = dict(
        data=x,
        graph=prepared,
        M=1,
        Tcycle=3,
        landmarks=100,
        splitting=8,
        graph_neighbors=12,
        knn_k=5,
        folds=4,
        backend="cpu",
        seed=19,
        progress=False,
        visual_init=False,
    )

    ordinary = kodama.matrix(**common)
    explicit_full = kodama.matrix(**common, _evolution_policy="full")
    no_transition = kodama.matrix(
        **common, _evolution_policy="no_transition_proposal"
    )

    np.testing.assert_array_equal(ordinary["res"], explicit_full["res"])
    np.testing.assert_array_equal(
        ordinary["best_labels"], explicit_full["best_labels"]
    )
    assert len(ordinary["run_diagnostics"]) == 1
    assert len(ordinary["cycle_diagnostics"]) == 3
    assert ordinary["run_diagnostics"][0]["cv_evaluations"] == 4
    for key in (
        "landmark_rows_hash",
        "initial_labels_hash",
        "fold_assignments_hash",
    ):
        assert ordinary["run_diagnostics"][0][key] == no_transition[
            "run_diagnostics"
        ][0][key]
    assert no_transition["run_diagnostics"][0]["transition_attempted"] == 0
    assert "_evolution_policy" not in inspect.signature(kodama.matrix).parameters
    with pytest.raises(ValueError, match="must be one of"):
        kodama.matrix(**common, _evolution_policy="not_a_policy")


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
    assert "handle" in graph and "indices" not in graph
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
    transition_fields = {
        "proposals_evaluated",
        "best_state_updates",
        "current_state_accepts",
        "stochastic_state_attempts",
        "stochastic_state_accepts",
        "current_state_rejections",
        "coarsening_moves",
        "absorption_moves",
    }
    assert transition_fields <= core_knn.keys()
    assert transition_fields <= core_pls.keys()
    for result in (core_knn, core_pls):
        assert result["proposals_evaluated"] == result["cycles_completed"]
        assert result["current_state_accepts"] + result["current_state_rejections"] == 0
        assert result["stochastic_state_accepts"] <= result["stochastic_state_attempts"]
    assert pca["scores"].shape == (60, 3)
    assert pca["loadings"].shape == (5, 3)
    assert pca["scores"].dtype == np.float32
    assert pca["precision"] == "float32"
    assert np.all(np.diff(pca["singular_values"]) <= 1e-5)
    assert kodama.graph_materialize(graph)["indices"].shape == (60, 5)
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
    assert emb_default.backend == "cpu"
    assert emb_default.optimizer == "csr_epoch_schedule"
    assert emb_default.graph_edges > 0
    assert emb_default.graph_max_weight == pytest.approx(1.0)
    assert emb_default.runtime_seconds >= 0.0
    assert emb_raw.initialization == "raw_pca"
    assert tsne_raw.initialization == "raw_pca"
    assert tsne_raw.backend == "cpu"
    assert tsne_raw.optimizer in {
        "opentsne_exact_sparse_knn_float32",
        "opentsne_fitsne_fft_grid_sparse_knn_float32",
    }
    assert tsne_raw.graph_edges == 0
    assert tsne_raw.graph_max_weight == 0.0
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

    materialized = kodama.graph_materialize(g)
    bare = {
        "indices": materialized["indices"],
        "distances": materialized["distances"],
    }
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
    from_raw = kodama.matrix(data=x, return_graph=True, **common)
    from_prepared = kodama.matrix(graph=g, **common)
    from_prepared_data = kodama.matrix(data=x, graph=g, **common)
    from_bare = kodama.matrix(graph=bare, **common)
    labels_only = kodama.matrix(data=x, **common)

    assert from_raw["graph_builds"] == 1
    assert from_prepared["graph_builds"] == 0
    assert from_prepared_data["graph_builds"] == 0
    assert from_bare["graph_builds"] == 0
    assert from_prepared["parameters"]["graph_uses_data_geometry"] is False
    assert from_prepared_data["parameters"]["graph_uses_data_geometry"] is True
    assert from_bare["parameters"]["graph_uses_data_geometry"] is False
    assert from_prepared_data["res"].shape == from_raw["res"].shape
    assert labels_only["knn"] is None
    np.testing.assert_array_equal(labels_only["res"], from_raw["res"])
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
    assert inspect.signature(kodama.CoreKNN).parameters["k"].default == 30
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
    assert matrix_signature.parameters["knn_k"].default == 30
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
        "folds",
        "visual_init",
        "progress",
        "apply_kodama_dissimilarity",
        "return_graph",
        ]
    assert "_evolution_policy" not in matrix_signature.parameters
    assert matrix_signature.parameters["spatial_resolution"].default == 0.4
    assert matrix_signature.parameters["spatial_constraint_mode"].default == "kmeans"
    assert matrix_signature.parameters["return_graph"].default is False
    assert "n_cores" in matrix_signature.parameters
    assert "n_threads" not in matrix_signature.parameters

    graph_signature = inspect.signature(kodama.matrix_graph)
    assert graph_signature.parameters["knn_k"].default == 30
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
        "folds",
        "visual_init",
        "progress",
        "apply_kodama_dissimilarity",
        "return_graph",
    ]
    assert "_evolution_policy" not in graph_signature.parameters
    assert graph_signature.parameters["spatial_resolution"].default == 0.4
    assert graph_signature.parameters["spatial_constraint_mode"].default == "kmeans"
    assert graph_signature.parameters["return_graph"].default is False
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
    assert visualization_signature.parameters["k"].default == 30
    assert "n_cores" in visualization_signature.parameters
    assert list(inspect.signature(kodama.graph).parameters) == [
        "data",
        "k",
        "metric",
        "backend",
        "n_cores",
            "gpu_device",
            "seed",
            "storage",
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


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal requires macOS")
def test_native_metal_umap_and_opentsne():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(96, 8)).astype(np.float32)
    try:
        layout = kodama.visualization(
            x,
            method="UMAP",
            k=10,
            n_epochs=10,
            backend="metal",
            seed=4,
        )
    except RuntimeError as error:
        if "No Apple Metal device is available" not in str(error):
            raise
        pytest.skip("the current process cannot access a Metal device")
    assert layout.shape == (96, 2)
    assert layout.initialization == "raw_pca"
    assert layout.initialization_backend == "metal"
    assert layout.backend == "metal"
    assert layout.optimizer == "metal_clean_atomic_edge_sampler"
    assert layout.graph_edges > 0
    assert layout.graph_max_weight == pytest.approx(1.0)
    assert layout.runtime_seconds >= 0.0
    assert np.all(np.isfinite(layout))
    tsne = kodama.visualization(
        x,
        method="opentsne",
        k=10,
        perplexity=3,
        early_exaggeration_iter=2,
        n_iter=3,
        backend="metal",
        seed=4,
    )
    assert tsne.shape == (96, 2)
    assert tsne.initialization == "raw_pca"
    assert tsne.initialization_backend == "metal"
    assert tsne.backend == "metal"
    assert tsne.optimizer == "metal_opentsne_fft_grid_sparse_knn_float32"
    assert tsne.graph_edges == 0
    assert tsne.graph_max_weight == 0.0
    assert tsne.runtime_seconds >= 0.0
    assert np.all(np.isfinite(tsne))
