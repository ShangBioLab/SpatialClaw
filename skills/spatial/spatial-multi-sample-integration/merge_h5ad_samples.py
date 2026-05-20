#!/usr/bin/env python3
"""Merge multiple spatial h5ad files for multi-sample integration.

This utility prepares a single merged AnnData with a batch column (default: sample)
so it can be used directly by spatial_multi_sample_integration.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge h5ad files for multi-sample integration")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input h5ad files, e.g. 151507/151507_raw.h5ad 151508/151508_raw.h5ad",
    )
    parser.add_argument("--output", required=True, help="Output merged h5ad path")
    parser.add_argument(
        "--batch-key",
        default="sample",
        help="obs column name used as batch key for integration (default: sample)",
    )
    parser.add_argument(
        "--sample-names",
        nargs="*",
        default=None,
        help=(
            "Optional sample names, one per input. "
            "If omitted, names are derived from parent folder or file stem."
        ),
    )
    parser.add_argument(
        "--join",
        choices=("inner", "outer"),
        default="inner",
        help="Gene join strategy when var differs across files (default: inner)",
    )
    return parser.parse_args()


def derive_sample_name(path: Path) -> str:
    parent = path.parent.name.strip()
    if parent:
        return parent
    return path.stem


def main() -> None:
    args = parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    if args.sample_names is not None and len(args.sample_names) > 0:
        if len(args.sample_names) != len(input_paths):
            raise ValueError("--sample-names must have the same length as --inputs")
        sample_names = args.sample_names
    else:
        sample_names = [derive_sample_name(p) for p in input_paths]

    if len(set(sample_names)) != len(sample_names):
        raise ValueError(f"Sample names must be unique, got: {sample_names}")

    adata_map: dict[str, ad.AnnData] = {}
    for sample_name, input_path in zip(sample_names, input_paths):
        adata = ad.read_h5ad(input_path)

        # Keep sample provenance explicit and make obs names globally unique.
        adata.obs[args.batch_key] = pd.Categorical([sample_name] * adata.n_obs)
        adata.obs_names = [f"{sample_name}_{idx}" for idx in adata.obs_names]
        adata.obs_names_make_unique()

        adata_map[sample_name] = adata

    merged = ad.concat(
        adata_map,
        axis=0,
        join=args.join,
        merge="same",
        label=args.batch_key,
        index_unique=None,
    )
    merged.obs[args.batch_key] = merged.obs[args.batch_key].astype("category")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output_path)

    print("Merge complete")
    print(f"Output: {output_path}")
    print(f"Shape: {merged.n_obs} cells x {merged.n_vars} genes")
    print(f"Batch key: {args.batch_key}")
    print(f"Batches: {merged.obs[args.batch_key].cat.categories.tolist()}")


if __name__ == "__main__":
    main()
