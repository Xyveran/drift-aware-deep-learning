from __future__ import annotations

import numpy as np
from typing import Generator, List, Tuple

#
# Validation
#


def _validate_caller_batch_args(
    available: set[int],
    train_set: set[int],
    test_set: set[int],
) -> None:
    """Guard used for splitting with caller-supplied batch lists.

    Raises:
        ValueError
            - Any train batch ID is absent from the data.
            - Any test batch ID is absent from the data.
            - Train and test sets overlap.
    """
    missing_train = train_set - available
    if missing_train:
        raise ValueError(
            f"train_batches {sorted(missing_train)} not found in batch_ids."
        )

    missing_test = test_set - available
    if missing_test:
        raise ValueError(f"test_batches {sorted(missing_test)} not found in batch_ids.")

    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"Batches {sorted(overlap)} appear in both train and test sets."
        )


#
# Batch-Based Split
#


def batch_based_split(
    batch_ids: np.ndarray, train_batches: List[int], test_batches: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices for a single static train/test split based on batch membership."""
    available = set(np.unique(batch_ids).tolist())

    _validate_caller_batch_args(available, set(train_batches), set(test_batches))

    train_idx = np.where(np.isin(batch_ids, train_batches))[0]
    test_idx = np.where(np.isin(batch_ids, test_batches))[0]

    return train_idx, test_idx


#
# Walk-Forward Splits
#


def walk_forward_splits(
    batch_ids: np.ndarray,
    n_test_batches: int = 1,
    min_train_batches: int = 1,
    step: int = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray, List[int], List[int]], None, None]:
    """Yield (train_idx, test_idx, train_batch_list, test_batch_list) for each walk-forward fold.

    Training window expands by `step` batches at each fold.
    Test window is a fixed-size block of `n_test_batches` following the training window.
    """
    all_batches = sorted(set(np.unique(batch_ids).tolist()))
    n_batches = len(all_batches)

    if min_train_batches + n_test_batches > n_batches:
        raise ValueError(
            f"Not enough batches ({n_batches}) for min_train_batches="
            f"{min_train_batches} + n_test_batches = {n_test_batches}."
        )

    train_end = min_train_batches

    while train_end + n_test_batches <= n_batches:
        train_batches = all_batches[:train_end]
        test_batches = all_batches[train_end : train_end + n_test_batches]

        train_idx = np.where(np.isin(batch_ids, train_batches))[0]
        test_idx = np.where(np.isin(batch_ids, test_batches))[0]

        yield train_idx, test_idx, train_batches, test_batches

        train_end += step


#
# Drift-Aware Split
#


def drift_aware_split(
    timestamps: np.ndarray,
    drift_timestamps: List[int],
    context_before: int = 0,
    recovery_window: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices where the test set is guaranteed to contain samples from just after
    each known drift event.

    Use when drift event timestamps are already known (e.g. from prior analysis or
    ground-truth labels) and you want to evaluate model recovery around those events.

    Parameters
        timestamps
            Per-window global timestamp array
        drift_timestamps
            Global timestamps at which drift events are known to occur
        context_before
            Number of timestamps before each drift event to include in the test set
        recovery_window
            Number of timestamps after each drfit event to include in the test set
            Will match `evaluation.recovery_window` in base.yaml

    Returns
        train_idx : np.ndarray
            Indices of windows that don't overlap with any drift test windows.
        test_idx : np.ndarray
            Indices of windows within a drift test window.
    """
    if len(drift_timestamps) == 0:
        raise ValueError("drift_timestamps is empty")

    test_mask = np.zeros(len(timestamps), dtype=bool)

    for dt in drift_timestamps:
        window_mask = (timestamps >= dt - context_before) & (
            timestamps < dt + recovery_window
        )
        test_mask |= window_mask

    earliest_test_ts = timestamps[test_mask].min() if test_mask.any() else np.inf
    train_mask = (~test_mask) & (timestamps < earliest_test_ts)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(
            "Drift-aware split produced an empty training set."
            "Reduce context_before or recovery_window, or provide more data."
        )
    if len(test_idx) == 0:
        raise ValueError(
            "Drift-aware split produced an empty test set."
            "Check that drift_timestamps fall within the timestamp range of the data."
        )

    return train_idx, test_idx
