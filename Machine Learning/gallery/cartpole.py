"""On-policy trajectories and held-out evaluation for Gymnasium CartPole."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh())
        self.actor = nn.Linear(48, 2)
        self.critic = nn.Linear(48, 1)

    def forward(self, x):
        features = self.body(x)
        return self.actor(features), self.critic(features).squeeze(-1)


def rollout(net, envs, observations, running, steps=32):
    states = []
    actions = []
    logs = []
    values = []
    rewards = []
    terminations = []
    ends = []
    next_vals = []
    completed = []
    for _ in range(steps):
        x = torch.as_tensor(np.stack(observations), dtype=torch.float32)
        with torch.no_grad():
            logits, value = net(x)
            dist = Categorical(logits=logits)
            a = dist.sample()
        batch = [env.step(int(action)) for env, action in zip(envs, a)]
        following, r, terminated, truncated, _ = zip(*batch)
        ended = np.logical_or(terminated, truncated)
        with torch.no_grad():
            _, nv = net(torch.as_tensor(np.stack(following), dtype=torch.float32))
        states.append(x)
        actions.append(a)
        logs.append(dist.log_prob(a))
        values.append(value)
        next_vals.append(nv)
        rewards.append(torch.tensor(r, dtype=torch.float32))
        terminations.append(torch.tensor(terminated))
        ends.append(torch.tensor(ended))
        for i, env in enumerate(envs):
            running[i] += r[i]
            if ended[i]:
                completed.append(running[i])
                running[i] = 0.0
                observations[i], _ = env.reset()
            else:
                observations[i] = following[i]
    # A time-limit truncation bootstraps from its final observation, but the trace
    # stops at either boundary so returns never span two episodes.
    r, term, end, v, nv = map(torch.stack, (rewards, terminations, ends, values, next_vals))
    delta = r + 0.99 * (~term) * nv - v
    advantage = torch.zeros_like(delta)
    tail = torch.zeros(len(envs))
    for i in reversed(range(steps)):
        tail = delta[i] + 0.99 * 0.95 * (~end[i]) * tail
        advantage[i] = tail
    return (
        torch.cat(states),
        torch.cat(actions),
        torch.cat(logs),
        advantage.flatten(),
        (advantage + v).flatten(),
    ), completed


def train_on_policy(seed, update, rounds=76):
    from gallery.neural import seed_cpu

    seed_cpu(seed)
    envs = [gym.make("CartPole-v1") for _ in range(8)]
    obs = [env.reset(seed=seed * 100 + i)[0] for i, env in enumerate(envs)]
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=0.0012)
    running = np.zeros(len(envs))
    history = []
    recent = []
    try:
        for _ in range(rounds):
            batch, completed = rollout(net, envs, obs, running)
            update(net, opt, batch)
            recent.extend(completed)
            history.append(np.mean(recent[-20:]) if recent else np.nan)
    finally:
        for env in envs:
            env.close()
    return net, history


@torch.no_grad()
def evaluate_cart(net, episodes=20):
    env = gym.make("CartPole-v1")
    lengths = []
    rng = np.random.default_rng(131)
    try:
        for i in range(episodes):
            state, _ = env.reset(seed=1700 + i)
            for length in range(1, 501):
                if net is None:
                    action = int(rng.integers(2))
                else:
                    logits, _ = net(torch.tensor(state))
                    action = int(logits.argmax())
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            lengths.append(length)
    finally:
        env.close()
    return np.array(lengths)


def cart_report(learn, unit):
    baseline = evaluate_cart(None)
    curves = []
    test = []
    print(f"Uniform-action mean return {baseline.mean():.1f}; SD {baseline.std(ddof=1):.1f}")
    for seed in (17, 29, 53):
        policy, curve = learn(seed)
        lengths = evaluate_cart(policy)
        curves.append(curve)
        test.append(lengths)
        print(
            f"Seed {seed}: mean return {lengths.mean():.1f}, median {np.median(lengths):.1f}, SD {lengths.std(ddof=1):.1f}, 500-step fraction {(lengths==500).mean():.1%}"
        )
    means = np.array(test).mean(1)
    print(f"Across training seeds: mean {means.mean():.1f}, sample SD {means.std(ddof=1):.1f}")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.1, 3.1), constrained_layout=True)
    for seed, curve in zip((17, 29, 53), curves):
        a.plot(np.arange(len(curve)) + 1, curve, lw=1.1, label=str(seed))
    a.set(xlabel=unit, ylabel="Return", title="Recent training episodes")
    a.legend(fontsize=8)
    b.boxplot([baseline, *test], tick_labels=["Random", "17", "29", "53"], showfliers=False)
    b.set(xlabel="Training seed", ylabel="Return", title="Held-out starting states")
