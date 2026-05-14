from __future__ import annotations

from pathlib import Path
from typing import Any

#
# Base Class
#


class Callback:
    """
    Base class for all training callbacks.

    Subclasses override only the hooks they need. The Trainer calls each hook
    at the appropriate point in the training loop, passing a shared ``state``
    dict that carries the current epoch, metrics, and model reference so
    callbacks can read and write training state without coupling directly
    to the Trainer.

    State keys populated by the Trainer
        epoch        : int     - current epoch (0-indexed)
        train_loss   : float   - average training loss for the epoch
        train_acc    : float   - training accuracy for the epoch
        val_loss     : float   - average validation loss for the epoch
        val_acc      : float   - validation accuracy for the epoch
        model        : nn.Module
        optimizer    : torch.optim.Optimizer
        stop_training: bool    - set True to trigger early stopping
    """

    def on_train_begin(self, state: dict[str, Any]) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...
    def on_epoch_begin(self, state: dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: dict[str, Any]) -> None: ...
    def on_batch_begin(self, state: dict[str, Any]) -> None: ...
    def on_batch_end(self, state: dict[str, Any]) -> None: ...


#
# Early Stopping
#


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
        monitor : str
            State key to monitor. Should match ``training.checkpoint_metric``
            in base.yaml (e.g. "val_loss").
        patience : int
            Number of epochs with no improvement before stopping.
        mode : str
            "min" if lower is better (loss), "max" if higher is better (accuracy).
        min_delta : float
            Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait: int = 0

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        current = state.get(self.monitor)
        if current is None:
            return

        improved = (
            current < self._best - self.min_delta
            if self.mode == "min"
            else current > self._best + self.min_delta
        )

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                state["stop_training"] = True


#
# Model Checkpoint
#


class ModelCheckpoint(Callback):
    """
    Save the model whenever the monitored metric improves.

    Parameters
        checkpoint_dir : Path
            Directory to write checkpoint files into.
        monitor : str
            State key to monitor.
        mode : str
            "min" or "max".
        filename : str
            Checkpoint filename. Defaults to "best_model.pt".
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        filename: str = "best_model.pt",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.path = Path(checkpoint_dir) / filename
        self.monitor = monitor
        self.mode = mode

        self._best: float = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        import torch

        current = state.get(self.monitor)
        if current is None:
            return

        improved = current < self._best if self.mode == "min" else current > self._best

        if improved:
            self._best = current
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": state["epoch"],
                    "model_state_dict": state["model"].state_dict(),
                    "optimizer_state_dict": state["optimizer"].state_dict(),
                    self.monitor: current,
                },
                self.path,
            )
