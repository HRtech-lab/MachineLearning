"""REINFORCE

Monte Carlo policy gradients reinforce sampled CartPole actions according to the discounted rewards that followed them. Each update uses six complete episodes.

Williams, Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (1992)
https://doi.org/10.1007/BF00992696
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


def train(seed):
    seed_cpu(seed)
    env = gym.make("CartPole-v1")
    net = ActorCritic()
    opt = torch.optim.Adam([*net.body.parameters(), *net.actor.parameters()], lr=0.003)
    history = []
    recent = []
    try:
        for batch in range(52):
            states = []
            actions = []
            returns = []
            for episode in range(6):
                state, _ = env.reset(seed=seed * 10000 + batch * 6 + episode)
                path = []
                while True:
                    obs = torch.tensor(state)
                    with torch.no_grad():
                        logits, _ = net(obs)
                        action = int(Categorical(logits=logits).sample())
                    states.append(obs)
                    actions.append(action)
                    state, r, done, cut, _ = env.step(action)
                    path.append(r)
                    if done or cut:
                        break
                recent.append(sum(path))
                g = 0.0
                discounted = []
                for r in reversed(path):
                    g = r + 0.99 * g
                    discounted.append(g)
                returns.extend(reversed(discounted))
            weight = torch.tensor(returns, dtype=torch.float32)
            weight = (weight - weight.mean()) / (weight.std() + 1e-6)
            logits, _ = net(torch.stack(states))
            loss = -(Categorical(logits=logits).log_prob(torch.tensor(actions)) * weight).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            history.append(np.mean(recent[-20:]))
    finally:
        env.close()
    return net, history


def main():
    cart_report(train, "Batch of six episodes")
    plt.show()


if __name__ == "__main__":
    main()
