from __future__ import annotations
from typing import Literal, NamedTuple
import torch
from torch import nn
from torch.nn import functional as F


class FlowStepOutput(NamedTuple):
    z: torch.Tensor
    log_abs_det: torch.Tensor


class FlowOutput(NamedTuple):
    z: torch.Tensor
    log_det_sum: torch.Tensor
    log_dets: torch.Tensor


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        flow_length: int,
        flow_type: Literal["planar", "radial"] = "planar",
    ):
        super().__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        flow_cls = {
            "planar": PlanarFlow,
            "radial": RadialFlow,
        }[flow_type]
        self.flows = nn.ModuleList([flow_cls(dim) for _ in range(flow_length)])

    def forward(self, z: torch.Tensor) -> FlowOutput:
        log_dets = []
        log_det_sum = z.new_zeros(z.shape[:-1])

        for flow in self.flows:
            step_output = flow(z)
            z = step_output.z
            log_dets.append(step_output.log_abs_det)
            log_det_sum = log_det_sum + step_output.log_abs_det

        return FlowOutput(
            z=z,
            log_det_sum=log_det_sum,
            log_dets=torch.stack(log_dets, dim=0),
        )


class PlanarFlow(nn.Module):
    """Planar flow with u_hat reparameterization for invertibility."""

    def __init__(self, dim: int, init_std: float = 1e-2, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.w = nn.Parameter(torch.empty(dim))
        self.u = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.empty(1))
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.uniform_(self.w, -init_std, init_std)
        nn.init.uniform_(self.u, -init_std, init_std)
        nn.init.uniform_(self.b, -init_std, init_std)

    def _u_hat(self) -> torch.Tensor:
        wu = torch.dot(self.w, self.u)
        m_wu = -1.0 + F.softplus(wu)
        w_norm_sq = self.w.square().sum().clamp_min(self.eps)
        return self.u + ((m_wu - wu) * self.w / w_norm_sq)

    def forward(self, z: torch.Tensor) -> FlowStepOutput:
        u_hat = self._u_hat()
        activation = F.linear(z, self.w.unsqueeze(0), self.b)
        h = torch.tanh(activation)
        z_next = z + h * u_hat
        h_prime = 1.0 - h.square()
        psi = h_prime * self.w
        det_jacobian = 1.0 + torch.sum(psi * u_hat, dim=-1)
        log_abs_det = torch.log(det_jacobian.abs().clamp_min(self.eps))
        return FlowStepOutput(z=z_next, log_abs_det=log_abs_det)


class RadialFlow(nn.Module):
    """Radial flow with alpha > 0 and beta > -alpha reparameterization."""

    def __init__(self, dim: int, init_std: float = 1e-2, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.z0 = nn.Parameter(torch.empty(dim))
        self.raw_alpha = nn.Parameter(torch.empty(1))
        self.raw_beta = nn.Parameter(torch.empty(1))
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.uniform_(self.z0, -init_std, init_std)
        nn.init.uniform_(self.raw_alpha, -init_std, init_std)
        nn.init.uniform_(self.raw_beta, -init_std, init_std)

    def _alpha_beta(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.raw_alpha) + self.eps
        beta = -alpha + F.softplus(self.raw_beta)
        return alpha, beta

    def forward(self, z: torch.Tensor) -> FlowStepOutput:
        alpha, beta = self._alpha_beta()
        diff = z - self.z0
        r = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
        h = 1.0 / (alpha + r)
        h_prime = -h.square()
        beta_h = beta * h
        z_next = z + beta_h * diff
        term1 = 1.0 + beta_h
        term2 = 1.0 + beta_h + beta * h_prime * r
        log_abs_det = (
            (self.dim - 1) * torch.log(term1.abs().clamp_min(self.eps))
            + torch.log(term2.abs().clamp_min(self.eps))
        ).squeeze(-1)
        return FlowStepOutput(z=z_next, log_abs_det=log_abs_det)
