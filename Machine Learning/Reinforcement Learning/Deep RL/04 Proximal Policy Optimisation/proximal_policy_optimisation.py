"""Proximal Policy Optimisation

PPO reuses a batch of recent CartPole transitions. Clipping the probability ratio limits the incentive to move the policy too far from the one that collected the batch.

Schulman et al., Proximal Policy Optimization Algorithms (2017)
https://arxiv.org/abs/1707.06347
Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation
https://arxiv.org/abs/1506.02438
Farama Foundation, Gymnasium CartPole-v1
https://gymnasium.farama.org/environments/classic_control/cart_pole/"""

from pathlib import Path
import sys

start = Path(__file__).resolve().parent
for parent in (start, *start.parents):
    if (parent / "gallery" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Open this example inside the Machine Learning collection.")


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Categorical
from gallery.cartpole import ActorCritic, cart_report, train_on_policy
from gallery.neural import seed_cpu


def update(net, opt, batch):
    states, actions, old_logp, adv, returns = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    for epoch in range(4):
        for ids in torch.randperm(len(states)).split(64):
            logits, values = net(states[ids])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(actions[ids]) - old_logp[ids]).exp()
            raw = ratio * adv[ids]
            clipped = ratio.clamp(0.8, 1.2) * adv[ids]
            actor = -torch.minimum(raw, clipped).mean()
            critic = nn.functional.mse_loss(values, returns[ids])
            loss = actor + 0.45 * critic - 0.01 * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.7)
            opt.step()


def main():
    def train(seed):
        return train_on_policy(seed, update)

    cart_report(train, "Rollout update")
    plt.show()


if __name__ == "__main__":
    main()
