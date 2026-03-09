from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    size: int
    sequence_length: int
    text_len: int
    text_vocab_size: int
    obj_dim: int
    human_dim: int
    robot_dim: int
    graph_nodes: int
    graph_node_dim: int
    contact_distance: float


class SyntheticStormDataset(Dataset):
    """Synthetic dataset for STORM V1 smoke training.

    It generates coherent object-human-robot sequences with deterministic contact
    phases and simple failure/recovery labels, enabling fast end-to-end checks.
    """

    def __init__(self, cfg: SyntheticConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        t = self.cfg.sequence_length

        text_tokens = torch.randint(
            0,
            self.cfg.text_vocab_size,
            (self.cfg.text_len,),
            generator=self.generator,
            dtype=torch.long,
        )

        # Object trajectory: smooth linear + sinusoidal velocity pattern.
        time = torch.linspace(0.0, 1.0, t)
        obj_pos = torch.stack(
            [
                0.8 * time,
                0.2 * torch.sin(2.0 * torch.pi * time),
                1.0 + 0.1 * torch.cos(2.0 * torch.pi * time),
            ],
            dim=-1,
        )
        obj_vel = torch.gradient(obj_pos, dim=0)[0]
        obj_feat = torch.cat([obj_pos, obj_vel], dim=-1)
        obj_state = self._pad_feature(obj_feat, self.cfg.obj_dim)

        # Human follows object with lag + noise.
        human_pos = obj_pos.clone()
        human_pos[:, 0] -= 0.15
        human_pos[:, 1] += 0.05
        human_pos += 0.01 * torch.randn_like(human_pos, generator=self.generator)
        human_vel = torch.gradient(human_pos, dim=0)[0]
        human_feat = torch.cat([human_pos, human_vel], dim=-1)
        human_state = self._pad_feature(human_feat, self.cfg.human_dim)

        # Robot trajectory: embodiment-specific realization (different but close outcome).
        robot_core = torch.cat(
            [
                human_pos * torch.tensor([1.1, 0.9, 1.0]),
                human_vel * torch.tensor([0.8, 1.2, 1.0]),
            ],
            dim=-1,
        )
        robot_state = self._pad_feature(robot_core, self.cfg.robot_dim)

        # Contact phase is driven by hand-object distance.
        hand_obj_dist = torch.norm(human_pos - obj_pos, dim=-1)
        phase = (hand_obj_dist < self.cfg.contact_distance).long()

        # Outcome labels.
        if torch.any(phase > 0):
            t_c = torch.argmax(phase).float() / max(1, t - 1)
        else:
            t_c = torch.tensor(0.0)

        post_obj_state = obj_state[-1, :6]
        stability = 1.0 - torch.clamp(torch.abs(robot_state[-1, 1]), 0.0, 1.0)

        outcome = {
            "t_c": t_c,
            "contact_state": phase.float(),
            "post_obj_state": post_obj_state,
            "stability": stability,
        }

        # Failure + recovery labels.
        # 0 normal, 1 contact_loss, 2 object_drop, 3 unstable
        fail_label = torch.tensor(0, dtype=torch.long)
        if phase.sum() < 2:
            fail_label = torch.tensor(1, dtype=torch.long)
        if obj_pos[-1, 2] < 0.8:
            fail_label = torch.tensor(2, dtype=torch.long)
        if stability < 0.4:
            fail_label = torch.tensor(3, dtype=torch.long)

        recovery_label = torch.tensor(min(int(fail_label.item()), 3), dtype=torch.long)

        graph_nodes = torch.randn(
            self.cfg.graph_nodes,
            self.cfg.graph_node_dim,
            generator=self.generator,
        )
        adjacency = torch.eye(self.cfg.graph_nodes)

        sample = {
            "text_tokens": text_tokens,
            "obj_state": obj_state,
            "human_state": human_state,
            "robot_state": robot_state,
            "phase": phase,
            "outcome_t_c": outcome["t_c"],
            "outcome_contact": outcome["contact_state"],
            "outcome_post_obj": outcome["post_obj_state"],
            "outcome_stability": outcome["stability"],
            "failure_label": fail_label,
            "recovery_label": recovery_label,
            "graph_nodes": graph_nodes,
            "graph_adj": adjacency,
        }
        return sample

    @staticmethod
    def _pad_feature(feat: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = feat.shape[-1]
        if current_dim == target_dim:
            return feat
        if current_dim > target_dim:
            return feat[..., :target_dim]
        pad = torch.zeros(feat.shape[0], target_dim - current_dim)
        return torch.cat([feat, pad], dim=-1)
