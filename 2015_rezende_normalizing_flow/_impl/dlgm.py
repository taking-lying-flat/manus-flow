import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .flow import FlowOutput


LOG_2PI = math.log(2.0 * math.pi)


def diag_normal_log_prob(
    z: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """log N(z | mean, diag(exp(logvar)))."""

    return -0.5 * (
        LOG_2PI
        + logvar
        + (z - mean).square() / logvar.exp()
    ).sum(dim=-1)


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    """log N(z | 0, I)."""
    return -0.5 * (LOG_2PI + z.square()).sum(dim=-1)


def beta_schedule(
    step: int,
    beta_start: float = 0.01,
    warmup_steps: int = 10_000,
) -> float:
    """Paper-style annealing: beta_t = min(1, 0.01 + t / 10000)."""
    return min(1.0, beta_start + step / warmup_steps)


class ConditionalPlanarFlow(nn.Module):
    """Amortized planar flow using per-sample parameters from the encoder."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _u_hat(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        wu = (w * u).sum(dim=-1, keepdim=True)
        m_wu = -1.0 + F.softplus(wu)
        w_norm_sq = w.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return u + ((m_wu - wu) * w / w_norm_sq)

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> FlowOutput:
        log_dets = []
        log_det_sum = z.new_zeros(z.shape[0])

        for k in range(u.shape[1]):
            uk = u[:, k, :]
            wk = w[:, k, :]
            bk = b[:, k, :]

            u_hat = self._u_hat(uk, wk)
            activation = (z * wk).sum(dim=-1, keepdim=True) + bk
            h = torch.tanh(activation)
            z = z + h * u_hat

            h_prime = 1.0 - h.square()
            psi = h_prime * wk
            det_jacobian = 1.0 + (psi * u_hat).sum(dim=-1)
            log_abs_det = torch.log(det_jacobian.abs().clamp_min(self.eps))

            log_det_sum = log_det_sum + log_abs_det
            log_dets.append(log_abs_det)

        if log_dets:
            stacked_log_dets = torch.stack(log_dets, dim=0)
        else:
            stacked_log_dets = z.new_zeros((0, z.shape[0]))

        return FlowOutput(z=z, log_det_sum=log_det_sum, log_dets=stacked_log_dets)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        flow_length: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.flow_length = flow_length

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        if flow_length > 0:
            flow_param_dim = flow_length * (2 * latent_dim + 1)
            self.flow_head = nn.Linear(hidden_dim, flow_param_dim)
            nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.flow_head.bias)
        else:
            self.flow_head = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        h = self.backbone(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(min=-10.0, max=10.0)

        if self.flow_length == 0:
            return mean, logvar, None, None, None

        params = self.flow_head(h).view(
            x.shape[0],
            self.flow_length,
            2 * self.latent_dim + 1,
        )
        u = params[:, :, :self.latent_dim]
        w = params[:, :, self.latent_dim:2 * self.latent_dim]
        b = params[:, :, 2 * self.latent_dim:].contiguous()
        return mean, logvar, u, w, b


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class DLGMLossOutput:
    loss: torch.Tensor
    free_energy: torch.Tensor
    recon_nll: torch.Tensor
    kl_like: torch.Tensor
    log_q0: torch.Tensor
    log_pz: torch.Tensor
    log_px_given_z: torch.Tensor
    log_det_sum: torch.Tensor


class DLGMNF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        flow_length: int,
        image_shape: tuple[int, int, int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.flow_length = flow_length
        c, h, w = image_shape
        flat = c * h * w
        if flat != input_dim:
            raise ValueError(
                "image_shape C*H*W ({}) must equal input_dim ({})".format(
                    flat,
                    input_dim,
                )
            )
        self.image_shape = image_shape

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, flow_length)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.flow = ConditionalPlanarFlow()

    def log_px_given_z(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -F.binary_cross_entropy_with_logits(
            logits,
            x,
            reduction="none",
        ).sum(dim=-1)

    def forward(self, x: torch.Tensor, beta: float = 1.0) -> DLGMLossOutput:
        x = x.flatten(start_dim=1)
        mean, logvar, u, w, b = self.encoder(x)

        std = torch.exp(0.5 * logvar)
        z0 = mean + std * torch.randn_like(std)
        log_q0 = diag_normal_log_prob(z0, mean, logvar)

        if self.flow_length > 0:
            flow_output = self.flow(z0, u, w, b)
            zk = flow_output.z
            log_det_sum = flow_output.log_det_sum
        else:
            zk = z0
            log_det_sum = z0.new_zeros(z0.shape[0])

        logits = self.decoder(zk)
        log_px_given_z = self.log_px_given_z(logits, x)
        log_pz = standard_normal_log_prob(zk)
        log_p_joint = log_px_given_z + log_pz

        beta_t = torch.as_tensor(beta, device=x.device, dtype=x.dtype)
        free_energy = log_q0 - beta_t * log_p_joint - log_det_sum
        loss = free_energy.mean()

        return DLGMLossOutput(
            loss=loss,
            free_energy=free_energy.detach(),
            recon_nll=(-log_px_given_z).mean().detach(),
            kl_like=(log_q0 - log_pz - log_det_sum).mean().detach(),
            log_q0=log_q0.detach(),
            log_pz=log_pz.detach(),
            log_px_given_z=log_px_given_z.detach(),
            log_det_sum=log_det_sum.detach(),
        )

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decoder(z)
        probs = torch.sigmoid(logits)
        c, h, w = self.image_shape
        return probs.view(num_samples, c, h, w)
