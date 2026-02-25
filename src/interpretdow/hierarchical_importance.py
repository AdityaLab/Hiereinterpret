from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class Hierarchy:
    """Tree-like hierarchy with parent and children lookups."""

    parent_by_node: dict[str, str | None]
    children_by_node: dict[str, list[str]]

    @classmethod
    def from_parent_to_children(
        cls, parent_to_children: dict[str, list[str]]
    ) -> "Hierarchy":
        parent_by_node: dict[str, str | None] = {}
        children_by_node: dict[str, list[str]] = {}

        for parent, children in parent_to_children.items():
            children_by_node[parent] = list(children)
            parent_by_node.setdefault(parent, None)

            for child in children:
                if child in parent_by_node and parent_by_node[child] != parent:
                    raise ValueError(
                        f"Node '{child}' has multiple parents: "
                        f"'{parent_by_node[child]}' and '{parent}'."
                    )
                parent_by_node[child] = parent
                children_by_node.setdefault(child, [])

        return cls(parent_by_node=parent_by_node, children_by_node=children_by_node)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Hierarchy":
        with Path(path).open("r", encoding="utf-8") as f:
            hierarchy = json.load(f)
        if not isinstance(hierarchy, dict):
            raise ValueError("Hierarchy JSON must be an object of parent -> children.")
        return cls.from_parent_to_children(hierarchy)

    def children(self, node: str) -> list[str]:
        return self.children_by_node.get(node, [])

    def parent(self, node: str) -> str | None:
        return self.parent_by_node.get(node)

    def nodes(self) -> set[str]:
        return set(self.parent_by_node.keys()) | set(self.children_by_node.keys())


def propagate_importance_scores(
    hierarchy: Hierarchy,
    target_node: str,
    edge_importance_fn: Callable[[str, str], float],
) -> dict[str, float]:
    """
    Propagate importance scores over a hierarchy using BFS.

    Implements:
        score_child  = score_current * I(child, current)
        score_parent = score_current * I(current, parent)
    """
    if target_node not in hierarchy.nodes():
        raise ValueError(f"Unknown target node '{target_node}' in hierarchy.")

    scores: dict[str, float] = {}
    scores[target_node] = 1.0

    queue: deque[tuple[str, float]] = deque([(target_node, 1.0)])
    visited: set[str] = set()

    while queue:
        current_node, current_score = queue.popleft()

        for child in hierarchy.children(current_node):
            if child in visited:
                continue
            child_score = current_score * float(edge_importance_fn(child, current_node))
            scores[child] = child_score
            queue.append((child, child_score))

        parent = hierarchy.parent(current_node)
        if parent is not None and parent not in visited:
            parent_score = current_score * float(
                edge_importance_fn(current_node, parent)
            )
            scores[parent] = parent_score
            queue.append((parent, parent_score))

        visited.add(current_node)

    return scores


def make_pearson_edge_importance_fn(
    time_series: pd.DataFrame,
) -> Callable[[str, str], float]:
    """
    Build I(x, y) from absolute Pearson correlation in [0, 1].

    The input data frame is expected to have one column per node and rows over time.
    """

    def _edge_importance(source_node: str, target_node: str) -> float:
        source = time_series[source_node].astype(float)
        target = time_series[target_node].astype(float)
        corr = source.corr(target)
        if pd.isna(corr):
            return 0.0
        return float(abs(corr))

    return _edge_importance


def permutation_edge_importance(
    source_series: pd.Series,
    target_series: pd.Series,
    lag: int = 6,
    random_state: int = 7,
) -> float:
    """
    Estimate directed edge importance using permutation importance.

    We model target_t from lagged source values:
        X_t = [source_{t-1}, ..., source_{t-lag}]
    and compute model-agnostic permutation importance on held-out data.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1.")

    source = source_series.to_numpy(dtype=float)
    target = target_series.to_numpy(dtype=float)

    if len(source) != len(target):
        raise ValueError("source_series and target_series must have equal length.")
    if len(source) <= lag + 8:
        raise ValueError("Not enough samples for selected lag.")

    feature_columns = [source[lag - i - 1 : len(source) - i - 1] for i in range(lag)]
    x = np.column_stack(feature_columns)
    y = target[lag:]

    split = int(0.8 * len(y))
    if split < 10 or len(y) - split < 5:
        raise ValueError("Not enough train/test samples after lagging.")

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    model.fit(x_train, y_train)

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=15,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    raw_score = float(np.mean(importance.importances_mean))
    return max(0.0, raw_score)


def make_permutation_edge_importance_fn(
    time_series: pd.DataFrame,
    lag: int = 6,
    random_state: int = 7,
) -> Callable[[str, str], float]:
    """
    Build I(x, y) with directed, model-agnostic permutation importance.
    """

    def _edge_importance(source_node: str, target_node: str) -> float:
        return permutation_edge_importance(
            source_series=time_series[source_node],
            target_series=time_series[target_node],
            lag=lag,
            random_state=random_state,
        )

    return _edge_importance
