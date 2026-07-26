""" Variational autoencoder (AEVB / SGVB) following Kingma & Welling, 2013: https://arxiv.org/pdf/1312.6114
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLPEncoder(nn.Module):
    """Inference model q_phi(z|x): MLP mapping x -> hidden h (shared representation)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return torch.tanh(self.fc_hidden(x_flat))


class BernoulliMLPDecoder(nn.Module):
    """Generative model p_theta(x|z): Bernoulli likelihood; outputs pixel probabilities."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        image_shape: tuple,
    ):
        super().__init__()
        self.image_shape = image_shape  # (C, H, W)
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc_hidden(z))
        y = torch.sigmoid(self.fc_out(h))
        return y.view(-1, *self.image_shape)


class VAE(nn.Module):
    """
    AEVB model (Kingma & Welling, 2013): q_phi(z|x)=N(z; mu, diag(sigma^2)), p_theta(x|z) Bernoulli on pixels,
    p(z)=N(0,I). Train by minimizing BCE reconstruction + analytic KL (single-sample SGVB, L=1).
    """

    def __init__(
        self,
        image_shape=(1, 28, 28),
        hidden_dim=400,
        n_latent_features=20,
    ):
        super().__init__()
        c, h, w = image_shape
        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        self.n_latent_features = n_latent_features
        input_dim = c * h * w

        self.encoder = GaussianMLPEncoder(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_latent_features)
        self.fc_logvar = nn.Linear(hidden_dim, n_latent_features)
        self.decoder = BernoulliMLPDecoder(
            n_latent_features, hidden_dim, input_dim, image_shape
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # SGVB: z = mu + sigma * eps, eps ~ N(0,I); differentiable path for gradients through (mu, sigma).
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(mu)
        return mu + std * eps

    def _bottleneck(self, h: torch.Tensor):
        # Variational parameters of q_phi(z|x).
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sample(self, num_samples=64, device="cpu"):
        """
        Prior samples z ~ p(z)=N(0,I), then x ~ p_theta(x|z). KL(q_phi(z|x) || p_theta(z)) regularizes the
        encoder parameters phi, pushing the approximate posterior toward the prior N(0, I).
        """
        z = torch.randn(num_samples, self.n_latent_features, device=device)
        return self.decoder(z)

    def forward(self, x):
        # One forward pass for a single-sample ELBO estimate (AEVB / SGVB).
        h = self.encoder(x)
        z, mu, logvar = self._bottleneck(h)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # -E_q[log p(x|z)] (Bernoulli BCE) + KL(q(z|x)||p(z)); sum over batch for ELBO gradient.
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD, BCE + KLD
