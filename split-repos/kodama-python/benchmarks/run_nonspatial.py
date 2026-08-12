#!/usr/bin/env python3
"""Benchmark the non-spatial Python KODAMA pipeline from an NPZ matrix."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import kodama


def timed(call):
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="NPZ containing a two-dimensional data array")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", type=Path, default=Path("kodama_nonspatial.csv"))
    parser.add_argument("--backends", default="cpu")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--Tcycle", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=10_000_000)
    parser.add_argument("--splitting", type=int, default=None)
    parser.add_argument("--graph-k", type=int, default=100)
    parser.add_argument("--embedding-k", type=int, default=30)
    parser.add_argument("--knn-k", type=int, default=30)
    parser.add_argument("--ncomp", type=int, default=50)
    parser.add_argument("--n-cores", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        if "data" not in archive:
            raise ValueError("input NPZ must contain 'data'")
        data = np.ascontiguousarray(archive["data"], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("'data' must be a two-dimensional matrix")

    dataset = args.name or args.input.stem
    ncomp = min(args.ncomp, data.shape[1])
    rows = []
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        prepared, graph_seconds = timed(
            lambda: kodama.graph(
                data, k=args.graph_k, backend=backend,
                n_cores=args.n_cores, seed=args.seed,
            )
        )
        for classifier in ("knn", "pls_lda"):
            fit, matrix_seconds = timed(
                lambda: kodama.matrix(
                    data=data,
                    graph=prepared,
                    classifier=classifier,
                    backend=backend,
                    M=args.M,
                    Tcycle=args.Tcycle,
                    landmarks=args.landmarks,
                    splitting=args.splitting,
                    knn_k=args.knn_k,
                    ncomp=ncomp,
                    n_cores=args.n_cores,
                    seed=args.seed,
                    return_graph=True,
                )
            )
            for method in ("UMAP", "opentsne"):
                kwargs = {"k": args.embedding_k} if method == "UMAP" else {"perplexity": 30}
                _, visual_seconds = timed(
                    lambda: kodama.visualization(fit, method, backend=backend, **kwargs)
                )
                rows.append({
                    "dataset": dataset,
                    "samples": data.shape[0],
                    "variables": data.shape[1],
                    "backend": backend,
                    "classifier": classifier,
                    "visualization": method,
                    "M": args.M,
                    "Tcycle": args.Tcycle,
                    "graph_seconds": graph_seconds,
                    "matrix_seconds": matrix_seconds,
                    "visualization_seconds": visual_seconds,
                    "pipeline_seconds": graph_seconds + matrix_seconds + visual_seconds,
                    "best_accuracy": float(np.max(np.asarray(fit["acc"]))),
                    "median_classes": float(np.median(np.asarray(fit.class_counts))),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
