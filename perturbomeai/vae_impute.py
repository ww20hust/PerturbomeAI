"""Masked beta-VAE for imputing missing input features.

Routine clinical panels are rarely complete: different individuals have
different subsets of assays measured. Before scoring, PerturbomeAI fills the
gaps with a masked variational autoencoder trained as a denoising model.

Key idea (the masking trick):
    The encoder consumes an interleaved ``(mask, value)`` representation. A
    cell that is naturally missing and a cell we deliberately hide during
    training both look identical to the encoder: ``(mask=0, value=0)``. By
    randomly hiding a fraction of *observed* cells each step and asking the
    decoder to reconstruct them, the model learns to infer any feature from
    whichever features are present.

At inference time nothing is hidden: we encode the observed cells, take the
posterior mean, and decode. The decoder output fills the missing cells while the
observed cells are kept as measured, yielding a complete matrix to score.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _guard_macos_openmp() -> None:
    """Avoid the macOS duplicate-OpenMP segfault.

    On macOS, PyTorch ships its own libomp while scikit-learn / XGBoost link a
    separate OpenMP runtime; running multi-threaded torch CPU kernels in the
    same process then segfaults. The imputation network is tiny, so constraining
    torch to a single CPU thread on Darwin sidesteps the clash with no practical
    cost (it is irrelevant on Linux and on GPU).
    """
    if platform.system() == "Darwin":
        torch.set_num_threads(1)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_encoder_input(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Interleave ``(mask, value)`` per feature.

    Args:
        x: (B, D) input with NaN for naturally missing cells.
        active_mask: (B, D) bool, observed cells deliberately hidden this step.

    Returns:
        (B, 2D) tensor where, per feature, ``mask`` is 1 for cells that are
        observed AND not actively hidden, else 0; ``value`` is the value where
        mask is 1, else 0. Naturally missing and actively hidden cells are both
        ``(0, 0)``.
    """
    obs = ~torch.isnan(x)
    m_eff = (obs & ~active_mask).float()
    v_eff = torch.where(m_eff.bool(), x, torch.zeros_like(x))
    b, d = x.shape
    out = torch.empty(b, 2 * d, device=x.device, dtype=v_eff.dtype)
    out[:, 0::2] = m_eff
    out[:, 1::2] = v_eff
    return out


def sample_active_mask(
    obs: torch.Tensor,
    *,
    p_min: float = 0.0,
    p_max: float = 0.7,
    fixed_rate: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per row, hide a random fraction of observed cells.

    Each row draws a rate ``r`` (uniform in [p_min, p_max], or ``fixed_rate``)
    and hides ``floor(r * n_observed)`` of its observed cells, chosen at random.
    """
    b, d = obs.shape
    device = obs.device
    n_obs = obs.sum(dim=1)
    if fixed_rate is not None:
        r = torch.full((b,), float(fixed_rate), device=device)
    else:
        r = torch.rand(b, device=device, generator=generator) * (p_max - p_min) + p_min
    k = (r * n_obs.float()).floor().long()
    k = torch.minimum(k, n_obs.long())

    noise = torch.rand(b, d, device=device, generator=generator)
    noise = torch.where(obs, noise, torch.full_like(noise, 2.0))  # unobserved sort last
    ranks = noise.argsort(dim=1).argsort(dim=1)
    return obs & (ranks < k.unsqueeze(1))


class MaskedVAE(nn.Module):
    """Masked beta-VAE that reconstructs (and thus imputes) feature panels."""

    def __init__(
        self,
        n_features: int,
        *,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)

        enc_layers: list[nn.Module] = []
        prev = 2 * self.n_features
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, self.latent_dim)
        self.fc_logvar = nn.Linear(prev, self.latent_dim)

        dec_layers: list[nn.Module] = []
        prev = self.latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.decoder_body = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Linear(prev, self.n_features)

    def encode(self, x: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(build_encoder_input(x, active_mask))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_out(self.decoder_body(z))

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor):
        mu, logvar = self.encode(x, active_mask)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction from the posterior mean (no masking)."""
        was_training = self.training
        self.eval()
        zero_mask = torch.zeros_like(x, dtype=torch.bool)
        mu, _ = self.encode(x, zero_mask)
        x_hat = self.decode(mu)
        if was_training:
            self.train()
        return x_hat


@dataclass
class VAEConfig:
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128)
    beta: float = 0.5
    kl_warmup_epochs: int = 5
    epochs: int = 30
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-5
    train_mask_min: float = 0.0
    train_mask_max: float = 0.7
    dropout: float = 0.1
    seed: int = 42
    device: str = "auto"


def _masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return x_hat.new_zeros(())
    diff = (x_hat - torch.nan_to_num(x, nan=0.0)) * mask.float()
    return diff.pow(2).sum() / mask.float().sum().clamp_min(1.0)


def train_masked_vae(x: np.ndarray, config: VAEConfig | None = None) -> MaskedVAE:
    """Train a MaskedVAE on a feature matrix (NaN = missing).

    Reconstruction loss is computed on the deliberately hidden (active) cells;
    the KL term is annealed over ``kl_warmup_epochs`` up to ``beta``.
    """
    cfg = config or VAEConfig()
    _guard_macos_openmp()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    n, d = x_t.shape
    model = MaskedVAE(
        n_features=d,
        latent_dim=cfg.latent_dim,
        hidden_dims=list(cfg.hidden_dims),
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    model.train()
    for epoch in range(cfg.epochs):
        beta = cfg.beta * min(1.0, (epoch + 1) / max(1, cfg.kl_warmup_epochs))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed + epoch))
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = x_t[idx].to(device)
            obs = ~torch.isnan(xb)
            if not obs.any():
                continue
            active = sample_active_mask(
                obs, p_min=cfg.train_mask_min, p_max=cfg.train_mask_max, generator=gen
            )
            x_hat, mu, logvar = model(xb, active)
            rec = _masked_mse(x_hat, xb, active)
            kl = model.kl_loss(mu, logvar)
            loss = rec + beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def impute(model: MaskedVAE, x: np.ndarray, *, batch_size: int = 8192, device: str = "auto") -> np.ndarray:
    """Return a dense matrix: observed cells kept, missing cells filled by the VAE."""
    _guard_macos_openmp()
    dev = resolve_device(device)
    model = model.to(dev)
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    n = x_t.shape[0]
    out = np.array(x, dtype=np.float32, copy=True)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x_t[start:end].to(dev)
        x_hat = model.reconstruct(xb).cpu().numpy()
        block = out[start:end]
        miss = ~np.isfinite(block)
        block[miss] = x_hat[miss]
        out[start:end] = block
    return out


def train_and_impute(x: np.ndarray, config: VAEConfig | None = None) -> tuple[np.ndarray, MaskedVAE]:
    """Convenience wrapper: train on ``x`` then impute its missing cells."""
    cfg = config or VAEConfig()
    model = train_masked_vae(x, cfg)
    filled = impute(model, x, batch_size=max(2048, cfg.batch_size), device=cfg.device)
    return filled, model
