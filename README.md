# CHL_LiquidMixture
Repository for training multicomponent liquid mixture with contrastive Hebbian learning. 
Source code and trained results are provided. 

The .ipynb notebooks can be uploaded directly to Google Colab and run as-is, with no extra files needed — they pull all required data directly from this GitHub repo. This is the easiest way to try the code without worrying about dependencies.

Jupyter notebooks for each paper section can be found as follows:
- Section 2.2: Target_Phase_Retrieval/Training_5memories32components_06012026.ipynb
- Section 2.3: Classifier/AND.ipynb, Classifier/Circular.ipynb, Classifier/Linear.ipynb, Classifier/XOR.ipynb, IO_Relations/Training_IO.ipynb
- Section 2.4: Classifier/MNIST_training.ipynb
- Section 3.1: Spatial_Phase/Training_5memories32components_spatial_06012026.ipynb

## Key simulation parameters

### Model A (non-spatial, single-compartment) dynamics — `forward_sim_x_ssolvent_clamp` (`Dynamics/Model_A.py`)

| Parameter | Description | Used in paper (classifier/IO sims) |
|---|---|---|
| `chi` | Flory-Huggins interaction matrix | — |
| `mu` | Chemical-potential offset per component | — |
| `clamp` | Number of leading components held fixed (zero mobility), e.g. to clamp classifier inputs | — |
| `t_end` | Total integration time | 300.0 |
| `dt` | Initial step size passed to the Dopri5 adaptive solver | 1e-1 |
| `max_steps` | Max solver steps allowed before diffrax raises an error | 30000 |

### Model B (spatial, Cahn-Hilliard) dynamics — `run_cahn_hilliard_multi_FH_rstab_jax_until_converged` (`Dynamics/Spatial_Cahn_Hilliard.py`)

| Parameter | Description | Used in paper (spatial sims) |
|---|---|---|
| `chi` | Flory-Huggins interaction matrix (nc x nc) | user-defined |
| `kappa` | Gradient-penalty (interfacial energy) coefficient per component | 5.0 |
| `M` | Per-component mobility | 1.0 |
| `r_stab` | Stabilization constant in the semi-implicit update; damps instability from strongly negative curvature in `chi` (larger = more stable, slower) | 2.0 |
| `dt` / `dt_min` / `dt_max` | Initial / floor / ceiling on the adaptive time step | 0.05 / 1e-6 / `dt` |
| `max_steps` | Max accepted steps before giving up | 2×10^5 |
| `max_time` | Max simulated time before giving up | 500.0 |
| `field_tol` | Convergence: stop once the relative per-step field change drops below this | 1e-6 |
| `energy_slope_tol` | Convergence: stop once the normalized energy-decrease rate drops below this fraction of the max observed slope | 1e-7 |

Please contact duyc190@caltech.edu for questions

Note: the chemical potential mu here is defined with an opposite sign compared to the paper.
