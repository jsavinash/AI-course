"""Generic, framework-agnostic training loop with callbacks.

Target layout calls for a reusable training loop. This reference
implementation is intentionally framework-free: it drives an epoch/step cycle
and invokes user-supplied callables plus optional callbacks. Apps keep their
own concrete training code; this is a shared convenience primitive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

TrainStep = Callable[[int, int], dict]  # (epoch, step) -> metrics dict
EpochHook = Callable[[int, list[dict]], None]  # (epoch, step_metrics) -> None


@dataclass
class Callback:
    on_epoch_start: Callable[[int], None] | None = None
    on_epoch_end: EpochHook | None = None


@dataclass
class TrainLoop:
    epochs: int
    steps_per_epoch: int
    train_step: TrainStep
    callbacks: list[Callback] = field(default_factory=list)
    verbose: bool = True

    def run(self) -> list[list[dict]]:
        history: list[list[dict]] = []
        for epoch in range(self.epochs):
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs}")
            if self.callbacks:
                for cb in self.callbacks:
                    if cb.on_epoch_start:
                        cb.on_epoch_start(epoch)
            step_metrics: list[dict] = []
            for step in range(self.steps_per_epoch):
                metrics = self.train_step(epoch, step)
                step_metrics.append(metrics)
                if self.verbose:
                    print(f"  step {step + 1}: {metrics}")
            for cb in self.callbacks:
                if cb.on_epoch_end:
                    cb.on_epoch_end(epoch, step_metrics)
            history.append(step_metrics)
        return history
