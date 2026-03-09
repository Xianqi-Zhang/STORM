from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from storm.training.losses import StormLoss


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    val_loss: float


class StormTrainer:
    def __init__(self, model: nn.Module, cfg: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = StormLoss(cfg)
        self.optim = AdamW(
            self.model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        self.grad_clip = cfg["train"]["grad_clip"]

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[TrainState]:
        history: list[TrainState] = []
        for epoch in range(1, self.cfg["train"]["epochs"] + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)
            history.append(TrainState(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
            print(f"[Epoch {epoch:03d}] train={train_loss:.4f} val={val_loss:.4f}")
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0

        for batch in loader:
            batch = self._to_device(batch)
            with torch.set_grad_enabled(training):
                pred = self.model(batch)
                losses = self.criterion(pred, batch)
                loss = losses["L_total"]

                if training:
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optim.step()

            total += loss.item()
            count += 1

        return total / max(count, 1)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}
