from __future__ import annotations

import torch
from torch import nn


class FailureRecoveryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, recovery_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.fail_cls = nn.Linear(hidden_dim, 4)
        self.recovery_cls = nn.Linear(hidden_dim, recovery_classes)

    def forward(self, hist_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.gru(hist_feat)
        h = out[:, -1]
        fail_logits = self.fail_cls(h)
        recovery_logits = self.recovery_cls(h)
        return {
            "fail_logits": fail_logits,
            "recovery_logits": recovery_logits,
        }
