from __future__ import annotations

import torch
from torch import nn


class CriticalWindowRefiner(nn.Module):
    def __init__(self, q_min: float, q_max: float, dq_max: float) -> None:
        super().__init__()
        self.q_min = q_min
        self.q_max = q_max
        self.dq_max = dq_max
        self.smoother = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.smoother.weight.fill_(1.0 / 3.0)

    def forward(self, robot_latent: torch.Tensor, phase_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        # robot_latent: [B, T, D]
        b, t, d = robot_latent.shape

        # Determine key windows from contact probability.
        contact_prob = torch.softmax(phase_logits, dim=-1)[..., 1]
        mask = (contact_prob > 0.5).float().unsqueeze(-1)

        q = torch.clamp(robot_latent, self.q_min, self.q_max)

        # light smoothing only on key windows
        q_s = q.reshape(b * d, t).unsqueeze(1)
        q_s = self.smoother(q_s).squeeze(1).reshape(b, d, t).transpose(1, 2)
        q_refined = torch.where(mask > 0, q_s, q)

        dq = q_refined[:, 1:] - q_refined[:, :-1]

        # Feasibility penalties
        q_violation = torch.relu(self.q_min - q_refined).mean() + torch.relu(q_refined - self.q_max).mean()
        dq_violation = torch.relu(torch.abs(dq) - self.dq_max).mean()
        intercept_feas = q_violation + dq_violation

        return {
            "q_refined": q_refined,
            "intercept_feas": intercept_feas,
        }
