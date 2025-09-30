# COSR: Connectivity-on-Shape Regression

***A Novel Bayesian Framework Uncovering Brain Connectivity-to-Shape Relationship in Preclinical Alzheimer’s Disease***

This repository contains the implementation of **COSR**, a Bayesian modeling framework designed to relate **brain connectivity patterns** (matrix-valued outcome) to **spatially varying structural (shape) features**. 
Implementations include full Bayesian MCMC (two variants) and a Variational Bayes alternative for scalable approximate inference.

Although motivated by applications to preclinical Alzheimer’s disease (AD), the methods are broadly applicable to other multimodal neuroimaging problems where connectivity-to-shape (or connectivity-to-image) relationships are of interest. 

---
## Repository structure

Top-level files and their purpose:

- `COSR_wrapper.m` — unified entry point that validates inputs and dispatches to the selected inference routine.
- `COSR_MCMC_BF.m` — MCMC sampler using a Bayesian factor model for correlated residuals.
- `COSR_MCMC_IND.m` — MCMC sampler assuming independent residuals.
- `COSR_VB_IND.m` — Variational Bayes / coordinate ascent implementation (scalable, approximate).
- `trandn.m` — truncated normal sampler used by MCMC routines.
- `example_simu/` — small scripts and helper functions for generating toy data and running examples.

---
## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Naomi-Ding/COSR.git
   cd COSR
   ```
2. Open MATLAB and add the COSR folder to your MATLAB path.
    ```matlab
    addpath('path_to_COSR');
    ```
3. Run the example simulation to generate toy data and run a small experiment:
   ```matlab
   example_simu/COSR_example_simu.m
   ```
4. Explore results (tables, and coefficient maps) in the `example_simu/` directory.
  The example is intentionally small so it runs quickly as a smoke test. Inspect variables in the workspace after the run, and result structures with estimates and diagnostics.
--- 


## Dependencies

- MATLAB (tested with R2019b – R2024a).
- Basic MATLAB toolboxes (Statistics, Linear Algebra). 
- No external packages are required for the example scripts included.


---

## Usage & Workflow
The main entry point is **``COSR_wrapper.m``**, which dispatches to one of the three COSR implementations based on the specified `error_model`. It accepts either positional arguments or a `params` struct (recommended for reproducibility). The wrapper handles input validation, default parameter settings, random initialization, and forwards to the chosen implementation.


### Data contract & calling patterns

COSR supports two common calling patterns. Provide either:

1) Spatial coordinates `Sv` and let the wrapper compute basis functions automatically:

  ```matlab
  COSR_wrapper(error_model, A, M, Sv, x, params)
  ```

2) Precompute basis functions and pass them in a `params` struct (in this case `Sv` may be `[]`):

```matlab
params.Psi = Psi; params.Lambda_sqrt = Lambda_sqrt; % precomputed
COSR_wrapper(error_model, A, M, [], x, params)
```

where the other core inputs are:
- `error_model` : string, one of {`'MCMC_BF'`, `'MCMC_IND'`, `'VB_IND'`} to select the inference method.
- `A` : (V, V, n) symmetric connectivity matrices for n subjects (zeros on diagonal).
- `M` : (n, S) matrix of shape measurements evaluated at S spatial locations (each row is a subject).
- `Sv` : (S, d) spatial coordinates of the S locations (e.g., mesh vertices (d=3) or 2D contours (d=2)) where shape is measured. Pass `[]` if providing `Psi` and `Lambda_sqrt` in `params`.
  1. If you pass `Sv`, the wrapper will compute an RBF kernel and extract eigenvectors/eigenvalues to form `Psi` and `Lambda_sqrt` (unless you override in `params`). 
  2. If you prefer to compute a custom basis externally (e.g., Laplace-Beltrami eigenfunctions), pass them via `params.Psi` and `params.Lambda_sqrt` and set `Sv = []`.
- `x` : (n, p) optional subject-level covariates (use `[]` when none).
- `params`: optional struct to control MCMC/VB settings, priors, and diagnostic output. The wrapper auto-fills sensible defaults when fields are omitted.
Note: 
  - `V` is the number of nodes in the connectivity matrix, 
  - `S` is the number of spatial locations (e.g., mesh vertices),  
  - `p` is the number of covariates (including intercept if desired).





*See the header of `COSR_wrapper.m` for more details on argument shapes and optional parameters.*



### Example: programmatic call (minimal)
- **Data preparation**: load or generate `A`, `M`, `x`, and either `Sv` or precompute `Psi` and `Lambda_sqrt`.
  ```matlab
  % generate toy data:
  seed = 2025;
  rng(seed);
  % Small settings for a quick smoke test
  V = 20;   % nodes (small)
  n = 40;   % subjects
  s = 6;    % image side length (s*s = S locations)
  S = s * s;
  p = 2; H = 3;
  % generator options (use the independent-error generator here)
  sigma2_e = 0.5; sigma2_gamma = 1; delta = 0.5; tau_pattern = 1;
  % Call the available data generator (simu_data_gen_2d)
  [A, M, x, Psi, Lambda_sqrt, B_true, BM, Gamma_x, tau_true, Z_true, w_true, ...
      gamma1_true, gamma2_true, sigma2_e_actual, SNR_A, SNR_BM, Sv] = ...
      simu_data_gen_2d(n, V, H, p, s, sigma2_e, sigma2_gamma, delta, ...
      tau_pattern, seed, false, 0.95, 20, 1);
  ```
- **Example A: provide spatial coordinates `Sv` and let wrapper compute `Psi`/`Lambda_sqrt`**
  ```matlab
  % Set random seed for reproducibility
  rng(123);
  params.error_model = 'MCMC_IND';
  params.nsamples = 2000; params.burnin = 500; params.thinning = 2;
  result = COSR_wrapper(params.error_model, A, M, Sv, x, params);
  estimates = result.estimates; samples = result.samples;
  ```
- **Example B: precompute basis functions and pass them via params (`Sv` optional / pass `[]`):**
  ```matlab
  % Set random seed for reproducibility
  rng(123);
  params.error_model = 'VB_IND';
  params.nsamples = 1000; 
  params.Psi = Psi; params.Lambda_sqrt = Lambda_sqrt;
  result = COSR_wrapper(params.error_model, A, M, [], x, params);
  estimates = result.estimates; 
  ```


### Inference options

- **``MCMC_BF``**: Full MCMC using a Bayesian factor model for correlated residuals — suited for moderate-sized problems and when modeling spatial residual correlation is important.
- **``MCMC_IND``**: MCMC assuming independent residuals — simpler and faster than BF variant.
- **``VB_IND``**: Variational Bayes / CAVI — fast, approximate inference for large-scale or high-dimensional problems.


### Outputs & interpretation

**Primary outputs produced by the COSR wrapper and inference functions:**
- `estimates` : point estimates (spatial coefficient maps, community assignments Z, effect sizes, etc.).
- `samples` : retained MCMC samples when using an MCMC method (for posterior summaries and credible intervals).
- `fitted_errors` : diagnostic objects from the example diagnostic utilities.

**Analyses typically focus on:**

- **Spatially varying coefficient maps ($\tau_{h,h'}(s)$)** that show where on the surface a given shape measurement is associated with the connectivity between subnetworks $h$ and $h'$.
- **Posterior inclusion probabilities (PIPs)** to identify the selection probabilities of associations, quantifying the uncertainty in the estimated effects.
- **Community / cluster assignments (Z)** which summarize modularity of nodes.

Visualize coefficient maps on your mesh/surface of choice (FreeSurfer, MATLAB patch/trisurf, or export to neuroimaging viewers).

---

## Example simulation (what the example does)

[`example_simu/COSR_example_simu.m`](example_simu/COSR_example_simu.m) performs a quick pipeline:

  1. Generates synthetic connectivity matrices `A` and shape observations `M` using `simu_data_gen_2d.m`.
  2. Splits data into training (80%), validation (10%), and testing (10%) sets.
  3. Runs a COSR routine on training data, based on chosen method (`MCMC_BF`, `MCMC_IND`, or `VB_IND`).
  4. Compares recovered vs. true coefficients /community assignments, computes performance evaluation metrics (MSE, Adjusted rand index (ARI), Recall, Precision, F1), and produces diagnostic summaries.

*Open the example script to change dimensions or noise settings for more stress testing.*

---
## Troubleshooting & tips

- Dimension mismatch errors: ensure `A` is (V,V,n) and symmetric; `M` must be (n,S). 
- Slow runs: reduce the number of basis functions `L` (fewer spatial basis) or use `VB_IND` for faster approximate inference.
- Reproducibility: set seed via `rng(2025)` before running simulations or MCMC.
- Memory: large meshes (S large) and many edges (V large) increase memory — only `VB_IND` is recommended for large-scale problems.



---

📌 Citation and licensing information can be added once the paper is published / accepted.
