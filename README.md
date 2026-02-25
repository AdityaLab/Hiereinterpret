# Hierarchical Industrial Demand Forecasting with Temporal and Uncertainty Explanations

*Appears in the 42nd IEEE International Conference on Data Engineering (ICDE) 2026*

## Setup

Requires Python `>=3.11`.

Install dependencies:

```bash
uv sync
```

Or:

```bash
pip install -e .
```

## Library API

Core implementation lives in `src/interpretdow/hierarchical_importance.py`:

- `Hierarchy.from_parent_to_children(...)`
- `propagate_importance_scores(...)`
- `make_permutation_edge_importance_fn(...)`
- `make_pearson_edge_importance_fn(...)`

The included interpretability method is **permutation importance**
(model-agnostic) to instantiate `I(x, y)` for time-series edges.

## Input Files


### Time-Series CSV

Add data to the dataset folder.

## Run

```bash
uv run python run_hierarchical_importance.py \
  --data-csv dataset/D1.csv \
  --hierarchy-json hierarchy.json \
  --target-node A1 \
  --method permutation \
  --lag 6 \
  --output-csv scores.csv
```

Alternative method:

```bash
uv run python run_hierarchical_importance.py \
  --data-csv dataset/D1.csv \
  --hierarchy-json hierarchy.json \
  --target-node A1 \
  --method pearson
```

The script prints ranked scores and optionally saves them to CSV.

# References

```bibtex
@inproceedings{kamarthi2026hierarchical,
  title={Hierarchical Industrial Demand Forecasting with Temporal and Uncertainty Explanations},
  author={Kamarthi, Harshavardhan and Xu, Shangqing and Tong, Xinjie and Zhou, Xingyu and Peters, James and Czyzyk, Joseph and Prakash, Aditya},
  booktitle={42nd IEEE International Conference on Data Engineering (ICDE)},
  year={2026}
}
```
