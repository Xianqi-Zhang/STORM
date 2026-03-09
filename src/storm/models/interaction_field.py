from __future__ import annotations

import torch
from torch import nn


class ObjectCentricInteractionField(nn.Module):
    def __init__(self, human_dim: int, obj_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lap_proj = nn.Sequential(
            nn.Linear(human_dim + obj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
        )

    def forward(self, human_seq: torch.Tensor, obj_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        # Use first 3 dims as position, next 3 dims as velocity.
        p_h = human_seq[..., :3]
        p_o = obj_seq[..., :3]
        v_h = human_seq[..., 3:6]
        v_o = obj_seq[..., 3:6]

        r_hand_obj = p_h - p_o
        v_rel = v_h - v_o

        lap_in = torch.cat([human_seq, obj_seq], dim=-1)
        g_lap = self.lap_proj(lap_in)

        return {
            "r_hand_obj": r_hand_obj,
            "v_rel": v_rel,
            "g_lap": g_lap,
        }
