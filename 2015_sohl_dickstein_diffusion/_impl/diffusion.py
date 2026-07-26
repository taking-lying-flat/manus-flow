from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from _impl.denoise import ConvDenseDenoiseNet


@dataclass(slots=True)
class DiffusionConfig:
    image_size: int = 32
    channels: int = 3
    trajectory_length: int = 1000
    n_temporal_basis: int = 10
    n_t_per_minibatch: int = 4

    # Floors / clamps for numerical safety
    step1_beta: float = 1e-3
    min_beta: float = 1e-6
    beta_eps: float = 1e-5
    variance_eps: float = 1e-20
    # Pixel quantization noise scale (paper §3.4)
    uniform_noise: float = 0.0

    # Denoiser network (multi-scale conv + dense regression head).
    n_hidden_conv: int = 64
    n_layers_conv: int = 4
    n_scales: int = 1
    n_hidden_dense_lower: int = 500
    n_hidden_dense_lower_output: int = 2
    n_hidden_dense_upper: int = 20
    n_layers_dense_lower: int = 4
    n_layers_dense_upper: int = 2


def _make_temporal_basis(trajectory_length: int, n_basis: int) -> torch.Tensor:
    """Gaussian-bump basis B in R^{T x K} used to make outputs time-aware."""
    if n_basis <= 1:
        return torch.ones(trajectory_length, 1)
    grid = torch.linspace(-1.0, 1.0, trajectory_length).view(-1, 1)
    centers = torch.linspace(-1.0, 1.0, n_basis).view(1, -1)
    width = (centers[0, 1] - centers[0, 0]).abs() / 2.0
    basis = torch.exp(-((grid - centers) ** 2) / (2.0 * width**2))
    return basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-12)


def extract(values: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Index into per-timestep schedule tensors and broadcast to image shape."""
    return values.gather(0, t.view(-1)).view(t.shape[0], *([1] * (len(target_shape) - 1)))


def gaussian_kl(
    q_mean: torch.Tensor, q_logvar: torch.Tensor,
    p_mean: torch.Tensor, p_logvar: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        p_logvar - q_logvar
        + (torch.exp(q_logvar) + (q_mean - p_mean) ** 2) * torch.exp(-p_logvar)
        - 1.0
    )


class LearnableBetaSchedule(nn.Module):
    """Learnable beta_t schedule parameterised through the temporal basis (paper §2.4)."""

    def __init__(
        self,
        trajectory_length: int,
        n_temporal_basis: int,
        step1_beta: float = 1e-3,
        min_beta: float = 1e-6,
        beta_eps: float = 1e-5,
    ):
        super().__init__()
        self.trajectory_length = int(trajectory_length)
        self.beta_eps = float(beta_eps)

        self.register_buffer("temporal_basis", _make_temporal_basis(trajectory_length, n_temporal_basis))
        min_betas = torch.full((trajectory_length,), float(min_beta))
        min_betas[0] += float(step1_beta)
        self.register_buffer("min_betas", min_betas)
        self.beta_perturb_coefficients = nn.Parameter(torch.zeros(n_temporal_basis))

    def forward(self) -> torch.Tensor:
        device = self.beta_perturb_coefficients.device
        dtype = self.beta_perturb_coefficients.dtype

        basis = self.temporal_basis.to(device=device, dtype=dtype)
        min_betas = self.min_betas.to(device=device, dtype=dtype)
        perturb = basis @ self.beta_perturb_coefficients
        baseline = (
            1.0 / torch.linspace(self.trajectory_length, 2.0, self.trajectory_length, device=device, dtype=dtype)
        ).clamp(self.beta_eps, 1.0 - self.beta_eps)

        beta_raw = torch.sigmoid(torch.logit(baseline) + perturb)
        betas = min_betas + beta_raw * (1.0 - min_betas - self.beta_eps)
        return betas.clamp(self.beta_eps, 1.0 - self.beta_eps)


class RegressionDenoiser(nn.Module):
    """Wraps the conv+dense regression net so it returns (mu_coeff, var_coeff)."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.n_temporal_basis = cfg.n_temporal_basis
        self.mlp = ConvDenseDenoiseNet(
            n_layers_conv=cfg.n_layers_conv,
            n_layers_dense_lower=cfg.n_layers_dense_lower,
            n_layers_dense_upper=cfg.n_layers_dense_upper,
            n_hidden_conv=cfg.n_hidden_conv,
            n_hidden_dense_lower=cfg.n_hidden_dense_lower,
            n_hidden_dense_lower_output=cfg.n_hidden_dense_lower_output,
            n_hidden_dense_upper=cfg.n_hidden_dense_upper,
            spatial_width=cfg.image_size,
            channels=cfg.channels,
            n_scales=cfg.n_scales,
            n_temporal_basis=cfg.n_temporal_basis,
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, temporal_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x_t.shape
        out = self.mlp(x_t).view(b, h, w, c, 2, self.n_temporal_basis)
        basis_t = temporal_basis.index_select(0, t).to(device=out.device, dtype=out.dtype)
        coeff = torch.einsum("bhwcpk,bk->bhwcp", out, basis_t)
        mu_coeff = coeff[..., 0].permute(0, 3, 1, 2).contiguous()
        var_coeff = coeff[..., 1].permute(0, 3, 1, 2).contiguous()
        return mu_coeff, var_coeff


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        config: DiffusionConfig,
        denoiser: nn.Module | None = None,
        schedule: LearnableBetaSchedule | None = None,
    ):
        super().__init__()
        self.config = config
        self.T = int(config.trajectory_length)
        self.channels = int(config.channels)
        self.variance_eps = float(config.variance_eps)
        self.uniform_noise = float(config.uniform_noise)
        self.n_t_per_minibatch = int(config.n_t_per_minibatch)

        self.schedule = schedule or LearnableBetaSchedule(
            trajectory_length=config.trajectory_length,
            n_temporal_basis=config.n_temporal_basis,
            step1_beta=config.step1_beta,
            min_beta=config.min_beta,
            beta_eps=config.beta_eps,
        )
        self.denoiser = denoiser or RegressionDenoiser(config)

    def schedule_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = self.schedule()
        alphas = 1.0 - betas
        return betas, alphas, torch.cumprod(alphas, dim=0)

    def add_uniform_noise(self, x0: torch.Tensor) -> torch.Tensor:
        if self.uniform_noise <= 0.0:
            return x0
        return x0 + (torch.rand_like(x0) - 0.5) * self.uniform_noise

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor,
        noise: torch.Tensor, alpha_bars: torch.Tensor,
    ) -> torch.Tensor:
        a = extract(alpha_bars, t, x0.shape)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise

    def q_posterior(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
        betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t = extract(betas, t, x_t.shape)
        alpha_t = extract(alphas, t, x_t.shape)
        alpha_bar_t = extract(alpha_bars, t, x_t.shape)
        alpha_bar_prev = (alpha_bar_t / alpha_t).clamp(0.0, 1.0)

        var_x0 = (1.0 - alpha_bar_prev).clamp_min(self.variance_eps)
        var_xt = (beta_t / alpha_t).clamp_min(self.variance_eps)
        posterior_var = (1.0 / (1.0 / var_x0 + 1.0 / var_xt)).clamp_min(self.variance_eps)
        posterior_mean = posterior_var * (
            torch.sqrt(alpha_bar_prev) * x0 / var_x0 + x_t / torch.sqrt(alpha_t) / var_xt
        )
        return posterior_mean, torch.log(posterior_var)

    def p_mean_log_variance(
        self, x_t: torch.Tensor, t: torch.Tensor,
        betas: torch.Tensor, alphas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_coeff, var_coeff = self.denoiser(x_t, t, self.schedule.temporal_basis)
        beta_t = extract(betas, t, x_t.shape)
        alpha_t = extract(alphas, t, x_t.shape)

        # 2015 reference implementation:
        # reverse mean is a perturbation around the forward-process mean.
        model_mean = torch.sqrt(alpha_t) * x_t + torch.sqrt(beta_t) * mu_coeff

        model_var = torch.sigmoid(
            torch.logit(beta_t.clamp(1e-6, 1.0 - 1e-6))
            + var_coeff / math.sqrt(self.T)
        ).clamp_min(self.variance_eps)
        return model_mean, torch.log(model_var)

    def kl_elbo_loss(
        self,
        q_mean: torch.Tensor, q_logvar: torch.Tensor,
        p_mean: torch.Tensor, p_logvar: torch.Tensor,
        betas: torch.Tensor, alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pixel negative log-likelihood bound (in bits/pixel),
        relative to a unit-variance Gaussian baseline.
        """
        kl = gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        const = torch.tensor(
            0.5 * (1.0 + math.log(2.0 * math.pi)), device=kl.device, dtype=kl.dtype,
        )

        # Boundary entropy terms: H(start) - H(end) + H(prior).
        beta_full = 1.0 - torch.exp(torch.log(alphas).sum())
        entropy_start = const + 0.5 * torch.log(betas[0].clamp_min(self.variance_eps))
        entropy_end = const + 0.5 * torch.log(beta_full.clamp_min(self.variance_eps))

        neg_log_bound = kl * self.T + entropy_start - entropy_end + const
        # Subtract entropy of N(0,1) baseline and convert nats -> bits.
        return ((neg_log_bound - const) / math.log(2.0)).mean() * self.channels

    def training_loss_once(self, x0: torch.Tensor) -> torch.Tensor:
        betas, alphas, alpha_bars = self.schedule_values()

        x0_used = self.add_uniform_noise(x0)
        t = torch.randint(1, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0_used)
        x_t = self.q_sample(x0_used, t, noise, alpha_bars)

        q_mean, q_logvar = self.q_posterior(x0_used, x_t, t, betas, alphas, alpha_bars)
        p_mean, p_logvar = self.p_mean_log_variance(x_t, t, betas, alphas)
        return self.kl_elbo_loss(q_mean, q_logvar, p_mean, p_logvar, betas, alphas)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        loss = x0.new_zeros(())
        for _ in range(self.n_t_per_minibatch):
            loss = loss + self.training_loss_once(x0)
        return loss / self.n_t_per_minibatch

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_size: int | tuple[int, int] | None = None,
        channels: int | None = None,
    ) -> torch.Tensor:
        """Reverse sampling: x_T ~ N(0, I), then iteratively x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        param = next(self.parameters())
        device, dtype = param.device, param.dtype

        channels = channels or self.config.channels
        if image_size is None:
            h = w = self.config.image_size
        elif isinstance(image_size, tuple):
            h, w = image_size
        else:
            h = w = int(image_size)

        x_t = torch.randn(batch_size, channels, h, w, device=device, dtype=dtype)
        betas, alphas, _ = self.schedule_values()
        betas, alphas = betas.to(device=device, dtype=dtype), alphas.to(device=device, dtype=dtype)

        # 2015 reference sampler: t = T-1 down to 1 (training only sees t >= 1).
        for step in range(self.T - 1, 0, -1):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            mean, logvar = self.p_mean_log_variance(x_t, t, betas, alphas)
            x_t = mean + torch.exp(0.5 * logvar) * torch.randn_like(x_t)
        return x_t


class DiffusionModel(GaussianDiffusion):
    """Thin adapter used by training; extra kwargs override DiffusionConfig fields."""

    def __init__(self, spatial_width: int, channels: int, uniform_noise: float = 0.0, **cfg_overrides):
        super().__init__(DiffusionConfig(
            image_size=int(spatial_width),
            channels=int(channels),
            uniform_noise=float(uniform_noise),
            **cfg_overrides,
        ))
