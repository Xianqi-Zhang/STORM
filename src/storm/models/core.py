from __future__ import annotations

import torch
from torch import nn


class OutcomeCentricCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        text_vocab_size: int,
        text_embed_dim: int,
        obj_dim: int,
        human_dim: int,
        robot_latent_dim: int,
        phase_classes: int,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)

        in_dim = text_embed_dim + obj_dim + human_dim + d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)

        self.human_head = nn.Linear(d_model, human_dim)
        self.obj_head = nn.Linear(d_model, obj_dim)
        self.robot_latent_head = nn.Linear(d_model, robot_latent_dim)
        self.phase_head = nn.Linear(d_model, phase_classes)

        # outcome head: t_c + post_obj(6) + stability(1)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        obj_state: torch.Tensor,
        human_state: torch.Tensor,
        embodiment_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, _ = obj_state.shape

        z_txt = self.text_embedding(text_tokens).mean(dim=1)  # [B, E]
        z_txt = z_txt.unsqueeze(1).repeat(1, t, 1)

        z_robot = embodiment_token.unsqueeze(1).repeat(1, t, 1)

        x = torch.cat([z_txt, obj_state, human_state, z_robot], dim=-1)
        x = self.input_proj(x)
        feat, _ = self.gru(x)

        human_pred = self.human_head(feat)
        obj_pred = self.obj_head(feat)
        robot_latent = self.robot_latent_head(feat)
        phase_logits = self.phase_head(feat)

        pooled = feat.mean(dim=1)
        outcome = self.outcome_head(pooled)

        return {
            "human_pred": human_pred,
            "obj_pred": obj_pred,
            "robot_latent": robot_latent,
            "phase_logits": phase_logits,
            "outcome_t_c": outcome[:, 0],
            "outcome_post_obj": outcome[:, 1:7],
            "outcome_stability": outcome[:, 7],
        }
