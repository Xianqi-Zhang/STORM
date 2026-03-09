from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class StormLoss:
    def __init__(self, cfg: dict) -> None:
        self.w = cfg["loss"]
        self.penetration_threshold = cfg["constraints"]["penetration_threshold"]
        self.contact_distance = cfg["constraints"]["contact_distance"]

    def __call__(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # L_motion
        l_motion = F.mse_loss(pred["human_pred"], batch["human_state"]) + F.mse_loss(pred["obj_pred"], batch["obj_state"])
        losses["L_motion"] = l_motion

        # L_interaction
        l_inter = F.cross_entropy(pred["phase_logits"].reshape(-1, pred["phase_logits"].shape[-1]), batch["phase"].reshape(-1))
        losses["L_interaction"] = l_inter

        # L_physics (simple proxies)
        dist = torch.norm(pred["human_pred"][..., :3] - pred["obj_pred"][..., :3], dim=-1)
        penetration = torch.relu(self.penetration_threshold - dist).mean()
        contact_target = (dist < self.contact_distance).float()
        contact_prob = torch.softmax(pred["phase_logits"], dim=-1)[..., 1]
        contact_consistency = F.mse_loss(contact_prob, contact_target)
        losses["L_physics"] = penetration + contact_consistency

        # L_robot
        robot_track = F.mse_loss(pred["q_refined"], batch["robot_state"])  # proxy for trackability
        losses["L_robot"] = robot_track

        # L_outcome
        l_tc = F.mse_loss(pred["outcome_t_c"], batch["outcome_t_c"])
        l_post = F.mse_loss(pred["outcome_post_obj"], batch["outcome_post_obj"])
        l_stab = F.mse_loss(torch.sigmoid(pred["outcome_stability"]), batch["outcome_stability"])
        losses["L_outcome"] = l_tc + l_post + l_stab

        # L_rel
        gt_r = batch["human_state"][..., :3] - batch["obj_state"][..., :3]
        gt_v = batch["human_state"][..., 3:6] - batch["obj_state"][..., 3:6]
        l_rel = F.mse_loss(pred["r_hand_obj"], gt_r) + F.mse_loss(pred["v_rel"], gt_v)
        losses["L_rel"] = l_rel

        # L_intercept_feas from refiner
        losses["L_intercept_feas"] = pred["intercept_feas"]

        # L_stab_post
        losses["L_stab_post"] = F.mse_loss(torch.sigmoid(pred["outcome_stability"]), batch["outcome_stability"])

        # Failure-aware losses
        losses["L_fail_cls"] = F.cross_entropy(pred["fail_logits"], batch["failure_label"])
        losses["L_recovery"] = F.cross_entropy(pred["recovery_logits"], batch["recovery_label"])

        total = (
            self.w["lambda_motion"] * losses["L_motion"]
            + self.w["lambda_interaction"] * losses["L_interaction"]
            + self.w["lambda_physics"] * losses["L_physics"]
            + self.w["lambda_robot"] * losses["L_robot"]
            + self.w["lambda_outcome"] * losses["L_outcome"]
            + self.w["lambda_rel"] * losses["L_rel"]
            + self.w["lambda_intercept_feas"] * losses["L_intercept_feas"]
            + self.w["lambda_stab_post"] * losses["L_stab_post"]
            + self.w["lambda_fail_cls"] * losses["L_fail_cls"]
            + self.w["lambda_recovery"] * losses["L_recovery"]
        )
        losses["L_total"] = total
        return losses
