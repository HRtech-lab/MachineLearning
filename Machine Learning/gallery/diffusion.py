"""A small epsilon-prediction model with DDPM and deterministic DDIM sampling.

Ho et al., DDPM
https://arxiv.org/abs/2006.11239
Song et al., DDIM
https://arxiv.org/abs/2010.02502
Nichol and Dhariwal, cosine noise schedule
https://arxiv.org/abs/2102.09672
"""

import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import pdist, cdist
from scipy.stats import wasserstein_distance


class NoiseSchedule:
    def __init__(self, steps=48):
        self.steps = steps
        t = torch.linspace(0, 1, steps + 1)
        curve = torch.cos((t + 0.008) / 1.008 * torch.pi / 2).square()
        curve = curve / curve[0]
        self.beta = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
        self.alpha = 1 - self.beta
        self.bar = self.alpha.cumprod(0)
        self.previous = torch.cat([torch.ones(1), self.bar[:-1]])

    def corrupt(self, clean, time, noise):
        alpha = self.bar[time, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


class NoisePredictor(nn.Module):
    def __init__(self, dimensions, width=128):
        super().__init__()
        self.register_buffer("frequency", torch.exp(-torch.arange(16) * np.log(10000.0) / 15))
        self.layers = nn.Sequential(
            nn.Linear(dimensions + 32, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, dimensions),
        )

    def forward(self, x, t):
        phase = t.float()[:, None] * self.frequency
        return self.layers(torch.cat([x, phase.cos(), phase.sin()], dim=1))


def train_noise(net, schedule, X, updates):
    opt = torch.optim.AdamW(net.parameters(), lr=0.0014)
    history = []
    for step in range(updates):
        sample = X[torch.randint(len(X), (112,))]
        t = torch.randint(schedule.steps, (len(sample),))
        noise = torch.randn_like(sample)
        estimate = net(schedule.corrupt(sample, t, noise), t)
        loss = nn.functional.mse_loss(estimate, noise)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        history.append(loss.item())
    net.eval()
    return np.array(history)


@torch.no_grad()
def heldout_noise(net, schedule, X):
    rng = torch.Generator().manual_seed(133)
    t = torch.randint(schedule.steps, (len(X),), generator=rng)
    eps = torch.randn(X.shape, generator=rng)
    err = net(schedule.corrupt(X, t, eps), t) - eps
    print(
        f"Held-out epsilon MSE {err.square().mean():.3f} | zero-predictor MSE {eps.square().mean():.3f}"
    )


@torch.no_grad()
def sample(net, schedule, initial, method, limit, count=None, seed=59):
    x = initial.clone()
    rng = torch.Generator().manual_seed(seed)
    if method == "ddpm":
        timeline = list(range(schedule.steps - 1, -1, -1))
    elif method == "ddim":
        timeline = np.linspace(schedule.steps - 1, 0, count or 12).round().astype(int).tolist()
        if len(set(timeline)) != len(timeline):
            raise ValueError("Sampling steps must be unique.")
    else:
        raise ValueError("Expected ddpm or ddim")
    for i, t in enumerate(timeline):
        bar = schedule.bar[t]
        eps = net(x, torch.full((len(x),), t, dtype=torch.long))
        clean = ((x - (1 - bar).sqrt() * eps) / bar.sqrt()).clamp(-limit, limit)
        if t == 0:
            x = clean
        elif method == "ddpm":
            prev = schedule.previous[t]
            mean = (
                prev.sqrt() * schedule.beta[t] * clean + schedule.alpha[t].sqrt() * (1 - prev) * x
            ) / (1 - bar)
            variance = schedule.beta[t] * (1 - prev) / (1 - bar)
            x = mean + variance.sqrt() * torch.randn(x.shape, generator=rng)
        else:
            prev = schedule.bar[timeline[i + 1]]
            consistent_eps = (x - bar.sqrt() * clean) / (1 - bar).sqrt()
            x = prev.sqrt() * clean + (1 - prev).sqrt() * consistent_eps
    return x.numpy()


def distribution_scores(reference, generated, bandwidth, directions=48):
    def k(a, b):
        return np.exp(-cdist(a, b, metric="sqeuclidean") / (2 * bandwidth**2))

    aa, bb = k(reference, reference), k(generated, generated)
    cross = k(reference, generated).mean()
    # Biased MMD² includes the diagonal, so identical empirical samples give zero.
    mmd2 = max(0.0, aa.mean() + bb.mean() - 2 * cross)
    rng = np.random.default_rng(73)
    vec = rng.normal(size=(reference.shape[1], directions))
    vec /= np.linalg.norm(vec, axis=0)
    a, b = reference @ vec, generated @ vec
    sw = np.mean([wasserstein_distance(a[:, i], b[:, i]) for i in range(directions)])
    return float(mmd2), float(sw)


def training_bandwidth(train):
    return max(float(np.median(pdist(train[:400]))), 1e-8)
