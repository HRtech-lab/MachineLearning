# Deep RL

DQN, REINFORCE, synchronous advantage actor–critic and PPO.

| Example | Question or method |
|---|---|
| [Deep Q-Network](01%20Deep%20Q-Network/deep_q_network.ipynb) · [Python](01%20Deep%20Q-Network/deep_q_network.py) | A neural network estimates four action values from a one-hot grid state. Experience replay and a delayed target network stabilise the bootstrap target. |
| [REINFORCE](02%20REINFORCE/reinforce.ipynb) · [Python](02%20REINFORCE/reinforce.py) | Monte Carlo policy gradients reinforce sampled CartPole actions according to the discounted rewards that followed them. Each update uses six complete episodes. |
| [Advantage Actor-Critic](03%20Advantage%20Actor-Critic/advantage_actor_critic.ipynb) · [Python](03%20Advantage%20Actor-Critic/advantage_actor_critic.py) | A synchronous actor–critic learns from eight CartPole environments. A value head estimates how much return to expect from each state. |
| [Proximal Policy Optimisation](04%20Proximal%20Policy%20Optimisation/proximal_policy_optimisation.ipynb) · [Python](04%20Proximal%20Policy%20Optimisation/proximal_policy_optimisation.py) | PPO reuses a batch of recent CartPole transitions. Clipping the probability ratio limits the incentive to move the policy too far from the one that collected the batch. |
