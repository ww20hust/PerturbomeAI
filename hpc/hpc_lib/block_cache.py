"""No-repeated-read block cache: write the feature matrix once, read it everywhere.

The shared imputed feature matrix ``X`` (N x F) is materialised once as a
float32 memmap. Every training job opens it read-only, so the heavy feature
matrix is never re-parsed or re-copied per locus. Loci are partitioned into
fixed-size blocks; each block stores its own label memmap (N x n_loci_block),
its locus id list, and a manifest ties everything together.
"""

from __future__ import annotations

import json

import numpy as np

from . import paths


def write_feature_memmap(x: np.ndarray) -> None:
    """Persist the shared feature matrix once as a float32 memmap."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, fdim = x.shape
    mm = np.memmap(paths.x_memmap_path(), dtype=np.float32, mode="w+", shape=(n, fdim))
    mm[:] = x[:]
    mm.flush()
    del mm
    with open(paths.x_shape_path(), "w", encoding="utf-8") as f:
        json.dump({"n": int(n), "n_features": int(fdim)}, f)


def open_feature_memmap() -> np.memmap:
    """Open the shared feature matrix read-only (no copy)."""
    with open(paths.x_shape_path(), encoding="utf-8") as f:
        shape = json.load(f)
    return np.memmap(
        paths.x_memmap_path(),
        dtype=np.float32,
        mode="r",
        shape=(shape["n"], shape["n_features"]),
    )


def write_block(block_id: int, labels: np.ndarray, locus_ids: list[str]) -> dict:
    """Write one block's label memmap + locus id list; return its manifest entry."""
    labels = np.ascontiguousarray(labels, dtype=np.int8)
    n, n_loci = labels.shape
    mm = np.memmap(paths.block_labels_path(block_id), dtype=np.int8, mode="w+", shape=(n, n_loci))
    mm[:] = labels[:]
    mm.flush()
    del mm
    with open(paths.block_labels_shape_path(block_id), "w", encoding="utf-8") as f:
        json.dump({"n": int(n), "n_loci": int(n_loci)}, f)
    with open(paths.block_locus_ids_path(block_id), "w", encoding="utf-8") as f:
        json.dump(list(locus_ids), f)
    return {"block_id": int(block_id), "n_loci": int(n_loci), "locus_ids": list(locus_ids)}


def open_block_labels(block_id: int) -> np.memmap:
    with open(paths.block_labels_shape_path(block_id), encoding="utf-8") as f:
        shape = json.load(f)
    return np.memmap(
        paths.block_labels_path(block_id),
        dtype=np.int8,
        mode="r",
        shape=(shape["n"], shape["n_loci"]),
    )


def load_block_locus_ids(block_id: int) -> list[str]:
    with open(paths.block_locus_ids_path(block_id), encoding="utf-8") as f:
        return json.load(f)


def build_block_cache(
    x: np.ndarray,
    labels: np.ndarray,
    locus_ids: list[str],
    *,
    block_size: int = 2000,
) -> dict:
    """Write the shared X memmap and partition loci into label-memmap blocks."""
    write_feature_memmap(x)
    n_loci = labels.shape[1]
    n_blocks = int(np.ceil(n_loci / block_size))
    entries = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = min((b + 1) * block_size, n_loci)
        entry = write_block(b, labels[:, lo:hi], locus_ids[lo:hi])
        entries.append(entry)
    manifest = {
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_loci": int(n_loci),
        "block_size": int(block_size),
        "n_blocks": int(n_blocks),
        "blocks": entries,
    }
    with open(paths.block_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest() -> dict:
    with open(paths.block_manifest_path(), encoding="utf-8") as f:
        return json.load(f)
