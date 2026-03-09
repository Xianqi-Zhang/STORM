from __future__ import annotations

import torch
from torch import nn

from .core import OutcomeCentricCore
from .embodiment import EmbodimentGraphEncoder
from .interaction_field import ObjectCentricInteractionField
from .refinement import CriticalWindowRefiner
from .recovery import FailureRecoveryHead


class StormV1(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        ccfg = cfg["constraints"]

        self.emb_encoder = EmbodimentGraphEncoder(
            node_dim=mcfg["graph"]["node_dim"],
            hidden_dim=mcfg["graph"]["hidden_dim"],
            out_dim=mcfg["d_model"],
            num_layers=mcfg["graph"]["num_layers"],
        )

        self.core = OutcomeCentricCore(
            d_model=mcfg["d_model"],
            text_vocab_size=mcfg["text_vocab_size"],
            text_embed_dim=mcfg["text_embed_dim"],
            obj_dim=mcfg["obj_dim"],
            human_dim=mcfg["human_dim"],
            robot_latent_dim=mcfg["robot_latent_dim"],
            phase_classes=mcfg["phase_classes"],
        )

        self.field = ObjectCentricInteractionField(
            human_dim=mcfg["human_dim"],
            obj_dim=mcfg["obj_dim"],
        )

        self.refiner = CriticalWindowRefiner(
            q_min=ccfg["q_min"],
            q_max=ccfg["q_max"],
            dq_max=ccfg["dq_max"],
        )

        self.recovery = FailureRecoveryHead(
            in_dim=mcfg["robot_latent_dim"] + 2,  # contact prob + obj visibility proxy
            hidden_dim=mcfg["d_model"],
            recovery_classes=mcfg["recovery_classes"],
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_robot = self.emb_encoder(batch["graph_nodes"], batch["graph_adj"])

        core_out = self.core(
            text_tokens=batch["text_tokens"],
            obj_state=batch["obj_state"],
            human_state=batch["human_state"],
            embodiment_token=z_robot,
        )

        field_out = self.field(core_out["human_pred"], core_out["obj_pred"])

        refine_out = self.refiner(core_out["robot_latent"], core_out["phase_logits"])

        contact_prob = torch.softmax(core_out["phase_logits"], dim=-1)[..., 1:2]
        obj_vis_proxy = (core_out["obj_pred"][..., 2:3] > 0.2).float()
        recovery_in = torch.cat([refine_out["q_refined"], contact_prob, obj_vis_proxy], dim=-1)
        recovery_out = self.recovery(recovery_in)

        return {
            **core_out,
            **field_out,
            **refine_out,
            **recovery_out,
        }
