from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from interpretdow import (
    Hierarchy,
    make_pearson_edge_importance_fn,
    make_permutation_edge_importance_fn,
    propagate_importance_scores,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute hierarchical node importance scores relative to a target."
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        required=True,
        help="CSV with one column per node and rows over time.",
    )
    parser.add_argument(
        "--hierarchy-json",
        type=Path,
        required=True,
        help="JSON mapping parent node to list of child nodes.",
    )
    parser.add_argument(
        "--target-node",
        type=str,
        required=True,
        help="Node ID to explain.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="permutation",
        choices=("permutation", "pearson"),
        help="Edge interpretability method for I(x, y).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=6,
        help="Lag length used only when --method=permutation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for scores.",
    )
    return parser


def _validate_columns(time_series: pd.DataFrame, hierarchy: Hierarchy) -> None:
    missing_nodes = sorted(node for node in hierarchy.nodes() if node not in time_series)
    if missing_nodes:
        missing = ", ".join(missing_nodes)
        raise ValueError(
            "The following hierarchy nodes are missing from CSV columns: "
            f"{missing}"
        )


def main() -> None:
    args = _build_parser().parse_args()

    time_series = pd.read_csv(args.data_csv)
    hierarchy = Hierarchy.from_json_file(args.hierarchy_json)
    _validate_columns(time_series=time_series, hierarchy=hierarchy)

    if args.method == "permutation":
        edge_importance_fn = make_permutation_edge_importance_fn(
            time_series=time_series, lag=args.lag
        )
    else:
        edge_importance_fn = make_pearson_edge_importance_fn(time_series=time_series)

    scores = propagate_importance_scores(
        hierarchy=hierarchy,
        target_node=args.target_node,
        edge_importance_fn=edge_importance_fn,
    )

    scores_df = (
        pd.DataFrame({"node": list(scores.keys()), "score": list(scores.values())})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    print(scores_df.to_string(index=False))

    if args.output_csv is not None:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved scores to {args.output_csv}")


if __name__ == "__main__":
    main()
