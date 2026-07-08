#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import kodama


def load_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def load_dataset(input_dir: Path, name: str):
    data = load_csv_matrix(input_dir / f"{name}_data.csv.gz")
    spatial = load_csv_matrix(input_dir / f"{name}_spatial.csv.gz")
    return data, spatial


def run_one(name, data, spatial, classifier, backend, args):
    print(f"[{name}] {classifier} backend={backend} M={args.M} Tcycle={args.Tcycle}", flush=True)
    start = time.perf_counter()
    result = kodama.matrix(
        data,
        spatial=spatial,
        M=args.M,
        Tcycle=args.Tcycle,
        ncomp=args.ncomp,
        landmarks=args.landmarks,
        knn_k=args.knn_k,
        classifier=classifier,
        backend=backend,
        progress=True,
    )
    elapsed = time.perf_counter() - start
    classes = np.array([len(np.unique(row)) for row in result["res"]])
    return {
        "dataset": name,
        "classifier": classifier,
        "backend": backend,
        "samples": data.shape[0],
        "variables": data.shape[1],
        "M": args.M,
        "Tcycle": args.Tcycle,
        "landmarks": args.landmarks,
        "knn_k": args.knn_k,
        "ncomp": args.ncomp,
        "runtime_seconds": elapsed,
        "core_runtime_seconds": float(result["runtime_seconds"]),
        "median_acc": float(np.median(result["acc"])),
        "max_acc": float(np.max(result["acc"])),
        "median_classes": float(np.median(classes)),
        "min_classes": int(np.min(classes)),
        "max_classes": int(np.max(classes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="exported-spatial")
    parser.add_argument("--out", default="kodama-python-br8100-merfish.csv")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--Tcycle", type=int, default=20)
    parser.add_argument("--landmarks", type=int, default=100000)
    parser.add_argument("--knn-k", type=int, default=30)
    parser.add_argument("--ncomp", type=int, default=50)
    parser.add_argument("--backends", default="cpu,cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for name in ("MERFISH", "Br8100"):
      data, spatial = load_dataset(input_dir, name)
      for backend in args.backends.split(","):
        for classifier in ("knn", "pls_lda"):
          row = run_one(name, data, spatial, classifier, backend, args)
          rows.append(row)
          with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
