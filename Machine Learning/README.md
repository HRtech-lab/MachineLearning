# Machine Learning

Each of the 118 examples has a runnable Python script and a notebook with its outputs saved. The first two examples fit a straight line to concrete strength and a logistic model to cell measurements. Both compare their test results with a simple baseline.

**Start with [linear regression](Supervised%20Learning/Regression/01%20Linear%20Regression/linear_regression.ipynb) or [logistic regression](Supervised%20Learning/Classification/01%20Logistic%20Regression/logistic_regression.ipynb).** For a visual introduction to clustering, [DBSCAN on two moons](Unsupervised%20Learning/Clustering/08%20DBSCAN/dbscan.ipynb) shows how a density-based method follows curved groups.

| Topic | Examples | What the notebooks measure |
|---|---:|---|
| [Regression](Supervised%20Learning/Regression/README.md) | 30 | Concrete strength across unseen recipes, plus a count forecast with a chronological test. |
| [Classification](Supervised%20Learning/Classification/README.md) | 27 | Cell measurements and handwritten digits, with class-wise errors and probability scores. |
| [Clustering](Unsupervised%20Learning/Clustering/README.md) | 11 | Blobs, ellipses, rings and two moons, with known generating labels for evaluation. |
| [Dimensionality reduction](Unsupervised%20Learning/Dimensionality%20Reduction/README.md) | 7 | Neighbour preservation and reconstruction on handwritten digits. |
| [Anomaly detection](Unsupervised%20Learning/Anomaly%20Detection/README.md) | 4 | Novelty scores for independent inliers and injected outliers. |
| [Computer vision · images](Computer%20Vision/Images/README.md) | 4 | Edges, gradient descriptors, convolutional classification and segmentation. |
| [Computer vision · videos](Computer%20Vision/Videos/README.md) | 4 | Sparse and dense motion, foreground extraction and clip classification. |
| [Diffusion models](Diffusion%20Models/README.md) | 3 | Stochastic and deterministic sampling on two moons, plus unconditional digit generation. |
| [Representation learning](Representation%20Learning/README.md) | 5 | JEPA, I-JEPA, V-JEPA, SimCLR and a denoising autoencoder, with frozen probes. |
| [RL · bandits](Reinforcement%20Learning/Bandits/README.md) | 3 | Exploration and regret with independent Bernoulli arms. |
| [RL · dynamic programming](Reinforcement%20Learning/Dynamic%20Programming/README.md) | 2 | Planning with the known transition model of a slippery grid. |
| [RL · sample-based control](Reinforcement%20Learning/Sample-Based%20Control/README.md) | 6 | Monte Carlo, temporal-difference and planning updates from sampled experience. |
| [RL · deep reinforcement learning](Reinforcement%20Learning/Deep%20RL/README.md) | 4 | DQN, REINFORCE, synchronous advantage actor–critic and PPO. |
| [Game theory](Game%20Theory/README.md) | 6 | Equilibrium solvers, learning in games, population dynamics and repeated play. |
| [Explainable AI](Explainable%20AI/README.md) | 2 | Exact SHAP contributions and the local fidelity and repeatability of LIME. |

## Run an example

Use Python 3.12 and start inside the cloned repository. The core dependencies cover the first regression, classification and clustering examples.

```sh
cd "Machine Learning"
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
python "Supervised Learning/Regression/01 Linear Regression/linear_regression.py"
```

On Windows, activate the environment with `.venv\Scripts\activate`. For the complete collection, including the neural networks, boosting libraries, explainers and notebook interface, install the full requirements.

```sh
python -m pip install -r requirements.txt
jupyter lab
```

The notebooks run from their own folders. Keep the `gallery` and `data` directories alongside the topic folders when cloning or downloading. The published runs used a CPU. Seed 17 is used for the main examples, with independent seeds where the experiment needs a separate test population or repeated training.

## How to read the scores

**Regression.** The main concrete comparison holds out whole recipes, including measurements at different curing ages. Each model also gets five recipe-grouped training folds. MAE and RMSE are in MPa. The residual charts use measured minus predicted strength.

**Classification.** The binary examples treat malignant as positive and reserve a stratified test set. The notebooks report class counts, precision, recall, F1, MCC, ROC AUC, average precision, Brier score and log loss. The multiclass digit examples report per-class and macro scores.

**Unsupervised and generative methods.** The clustering and embedding estimators fit coordinates or pixels. Generating labels score the clusters, and digit labels stratify the embedding sample and colour its plots. Diffusion examples compare independent samples using MMD² and sliced Wasserstein distance. The image examples include generated samples in their original order.

**Representation learning.** Raw inputs, a frozen random encoder and a frozen trained encoder use the same linear-probe protocol. The JEPA examples are small adaptations of the published architectures. For example, on the synthetic motion task the V-JEPA probe reached 0.338 accuracy, compared with 0.300 for the random encoder and 0.362 for raw inputs.

**Control and games.** The control examples report return against a random policy. The sampled control and deep RL notebooks repeat training with three seeds. Game theory uses incentives to deviate, regret or equilibrium gaps where those quantities apply.

## Data and licence

Each example cites its method and data. The example code uses the repository’s [MIT licence](../LICENSE). The concrete spreadsheet was converted to CSV and its columns renamed. The UCI datasets and their adaptations retain their respective attribution terms. [Data credits and changes](data/README.md)
