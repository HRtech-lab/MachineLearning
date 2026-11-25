# Diffusion Models

Stochastic and deterministic sampling on two moons, plus unconditional digit generation.

| Example | Question or method |
|---|---|
| [DDPM on Two Moons](01%20DDPM%20on%20Two%20Moons/ddpm_on_two_moons.ipynb) · [Python](01%20DDPM%20on%20Two%20Moons/ddpm_on_two_moons.py) | A noise predictor learns how to undo a forward diffusion process on two crescents. Running the full stochastic reverse chain turns normal noise into a new point cloud. |
| [DDIM Sampling](02%20DDIM%20Sampling/ddim_sampling.ipynb) · [Python](02%20DDIM%20Sampling/ddim_sampling.py) | DDIM can skip noise levels while using the same trained noise predictor. Compare twelve deterministic steps with forty-eight stochastic DDPM steps from the same initial noise. |
| [Digit Diffusion](03%20Digit%20Diffusion/digit_diffusion.ipynb) · [Python](03%20Digit%20Diffusion/digit_diffusion.py) | An unconditional diffusion model generates eight-by-eight handwritten digits. The denoiser is a small multilayer perceptron with timestep features. |
