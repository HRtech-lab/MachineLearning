# Sample-Based Control

Monte Carlo, temporal-difference and planning updates from sampled experience.

| Example | Question or method |
|---|---|
| [Monte Carlo Control](01%20Monte%20Carlo%20Control/monte_carlo_control.ipynb) · [Python](01%20Monte%20Carlo%20Control/monte_carlo_control.py) | First-visit Monte Carlo control estimates action values from complete sampled returns. Epsilon-soft exploration keeps more than one action in play. |
| [SARSA](02%20SARSA/sarsa.ipynb) · [Python](02%20SARSA/sarsa.py) | SARSA evaluates the next action actually selected by its exploratory behaviour policy. |
| [Expected SARSA](03%20Expected%20SARSA/expected_sarsa.ipynb) · [Python](03%20Expected%20SARSA/expected_sarsa.py) | Expected SARSA averages over the next epsilon-greedy action distribution instead of drawing a single bootstrap action. |
| [Q-Learning](04%20Q-Learning/q_learning.ipynb) · [Python](04%20Q-Learning/q_learning.py) | Q-learning bootstraps from the largest next-state action value while collecting experience with an exploratory policy. |
| [Double Q-Learning](05%20Double%20Q-Learning/double_q_learning.ipynb) · [Python](05%20Double%20Q-Learning/double_q_learning.py) | Double Q-learning uses one table to select a next action and the other to evaluate it. The table that receives the update is chosen at random. |
| [Dyna-Q](06%20Dyna-Q/dyna_q.ipynb) · [Python](06%20Dyna-Q/dyna_q.py) | Dyna-Q follows each real transition with five planning updates drawn from an empirical model of previously seen state–action pairs. |
