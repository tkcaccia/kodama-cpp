# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

import numpy as np

import kodama


def test_matrix_knn_and_pls_lda_cpu():
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
    assert knn.best_labels.shape == (90,)
    assert knn["best_labels"].shape == (90,)
    assert knn.class_counts.shape == (1,)
    assert knn["parameters"]["classifier"] == "knn"
    assert any(row["step"] == "runtime_seconds" for row in kodama.timing(knn))


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
    emb = kodama.visualization(
        graph,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
        graph_mode="binary",
    )
    emb_fuzzy = kodama.visualization(
        graph,
        method="UMAP",
        k=5,
        n_epochs=3,
        backend="cpu",
        graph_mode="fuzzy",
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
    assert emb.shape == (60, 2)
    assert emb_fuzzy.shape == (60, 2)
    assert np.all(np.isfinite(emb))
    assert np.all(np.isfinite(emb_fuzzy))
    assert clu["membership"].shape == (60,)


def test_matrix_graph_cpu():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 7)).astype(np.float32)
    g = kodama.graph(x, k=15, backend="cpu")

    knn = kodama.matrix_graph(
        g,
        M=1,
        Tcycle=1,
        landmarks=50,
        splitting=5,
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


def test_diagnostics():
    diag = kodama.diagnostics()
    assert "extension" in diag
    assert "CONDA_PREFIX" in diag["environment"]
