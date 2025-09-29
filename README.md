# COSR: Connectivity-on-Shape Regression

***A Novel Bayesian Framework Uncovering Brain Connectivity-to-Shape Relationship in Preclinical Alzheimer’s Disease***

This repository hosts the implementation of **COSR**, a Bayesian modeling framework designed to relate **brain connectivity patterns** to **spatially varying structural (shape) features**. The method is motivated by applications to preclinical Alzheimer’s disease (AD), but is broadly applicable to multimodal neuroimaging problems where connectivity-to-shape (or connectivity-to-geometry) relationships are of interest.

<!-- --- -->

<!-- ## Table of Contents
- [COSR: Connectivity-on-Shape Regression](#cosr-connectivity-on-shape-regression)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Quick Start](#quick-start)
  - [Usage \& Workflow](#usage--workflow)
    - [Example / Simulation](#example--simulation)
    - [Inference Options](#inference-options)
    - [Outputs \& Interpretation](#outputs--interpretation)
    - [Dependencies \& Environment](#dependencies--environment) -->
<!-- - [License & Citation](#license--citation)  
- [Contact](#contact)   -->

<!-- --- -->

<!-- ## Background & Motivation

In many neuroimaging studies, one seeks to understand how **functional connectivity networks** (e.g. edge weights between brain regions) relate to **structural brain properties**, such as regional volumes or local shape deformations. However:

- Traditional approaches often treat connectivity and structure separately.
- They may ignore spatial dependencies on structural surfaces or lack uncertainty quantification.
- A connectivity-to-shape mapping would allow one to localize *which points or subregions of a brain surface* are associated with particular connectivity edges.

COSR is designed to fill this methodological gap:

1. **Integrative modeling**: It regresses *pointwise structural shape deformation* on connectivity predictors.
2. **Spatial structure**: It encodes smoothness (spatial correlation) across the shape surface.
3. **Sparsity / Edge selection**: It allows automatic selection of which connectivity edges are relevant for each structural location.
4. **Uncertainty quantification**: Via Bayesian inference (MCMC or variational) one obtains credible intervals, posterior inclusion probabilities, etc.

In the Alzheimer’s setting, this helps to detect *early shape signatures* tied to connectivity alterations in preclinical disease. -->

<!-- ## Model & Methodology

At a high level, the COSR model posits, for each subject:

\[
y_i(\mathbf{s}) = \sum_{e} x_{i,e} \, \tau_e(\mathbf{s}) + \varepsilon_i(\mathbf{s}),
\]

where:

- \( y_i(\mathbf{s}) \) = structural shape deformation (or measure) at spatial location \(\mathbf{s}\) on a surface mesh,
- \( x_{i,e} \) = connectivity feature (edge) \(e\) for subject \(i\),
- \( \tau_e(\mathbf{s}) \) = spatially varying coefficient function for edge \(e\),
- \(\varepsilon_i(\mathbf{s})\) = noise/error term (spatial residual).

Key methodological features:

- **Hierarchical priors** on \(\tau_e(\mathbf{s})\) that encourage **local smoothness over \(\mathbf{s}\)** (e.g. via Gaussian Markov random fields or spatial kernels) and **edge-level sparsity** (e.g. spike-and-slab, shrinkage priors).
- **Posterior inference**:
  - **MCMC samplers** (for gold-standard full Bayesian inference),
  - **Variational / coordinate-ascent methods** (for scalability in larger data or high-dimensional settings).
- **Regularization / hyperparameter tuning** can be handled via empirical Bayes or cross-validation within the framework. -->


---

## Repository Structure
```bash
COSR/
│
├── example_simu/        # Scripts for reproducing toy simulations.
├── COSR_MCMC_BF.m       # MCMC under Bayesian Factor error structure
├── COSR_MCMC_IND.m      # MCMC under Independent error structure
├── COSR_VB_IND.m        # Variational Bayes / CAVI inference
├── COSR_wrapper.m       # Wrapper to call COSR variants
├── trandn.m             # Utility for truncated normal sampling in MCMC
└── README.md            # This file
```

---
## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/COSR.git
   cd COSR
   ```
2. Open MATLAB and add the project folder to your path.
3. Run the example simulation:
   ```matlab
   example_simu/COSR_example_simu.m
   ```
4. Explore results (tables, and coefficient maps) in the `example_simu/` directory.

--- 


## Usage & Workflow
The main entry point is `COSR_wrapper.m`, which dispatches to one of the three COSR implementations based on the specified `error_model`. The wrapper handles input validation, default parameter settings, and output formatting.

### Example / Simulation

**To run a toy simulation:**

1. Add the COSR folder to your MATLAB path. (e.g., `addpath('path_to_COSR')`).
2. Navigate to `example_simu/`.
3. Execute the provided script (e.g. `COSR_example_simu.m`) which:
   - Simulates synthetic connectivity \(A\) and shape predictor \(M_i(s)\),
   - Randomly splits data into training (80%), validation (10%), and testing (10%) sets,
   - Runs COSR inference on training data, based on chosen method (MCMC_BF, MCMC_IND, or VB),
   - Compares recovered vs. truth shape-FC coefficients \(\tau_{h,h'}(s)\), and community membership \(Z\).
   - Produces performance metrics (MSE, Adjusted rand index (ARI), Recall, Precision, F1).
4. Examine output summary tables.

<!-- This lets users test that the pipeline is working before applying to real data. -->

---

<!-- ## Real Data Application

To analyze a real study (e.g. using ADNI or A4 data):

1. Prepare connectivity matrices and shape measurements for all subjects.
2. Organize into compatible MATLAB format (e.g. `.mat` files).
3. Call `COSR_wrapper.m` or the specific inference function, passing:
   - `A` (connectivity matrix)
   - `M(s)` (shape predictor)
   - Hyperparameter settings
   - Inference choice (MCMC_BF, MCMC_IND, or VB),
4. Obtain outputs: coefficient estimates, community membership, PIPs, credible intervals, etc.
5. Visualize effects using appropriate mesh plotting tools (e.g. connect with FreeSurfer surface, or export to external viewers).

--- -->

### Inference Options

- **MCMC (BF / IND)**  
  - Full Bayesian inference, slower but usually more accurate uncertainty quantification.
  - Useful for low dimensions/sample sizes.
  <!-- - Slower but more accurate, better to explore full posterior uncertainty. -->
  <!-- - Useful for moderate dimension / sample size. -->

- **Variational Bayes / CAVI (VB, IND)**  
  - Faster, scalable inference via mean-field approximation and coordinate ascent variational inference.
  - Useful for high-dimensional settings or large-scale data.
  <!-- - May underestimate posterior variance; good for initial screening. -->

<!-- - **Hyperparameter tuning**  
  - Use cross-validation, For slab variances or smoothing strength, one can use cross-validation, empirical Bayes, or holdout likelihood. -->

<!-- - **Convergence diagnostics**  
  - For MCMC: monitor trace plots, effective sample size, Gelman–Rubin diagnostics.
  - For VB: monitor ELBO progression.

- **Memory & computation**  
  - The method may be expensive for large meshes or many edges—consider reducing basis dimension or preselecting edges. -->

---

### Outputs & Interpretation

**Key outputs from COSR include:**

- **Estimated spatially varying coefficient maps** \(\hat{\tau}_{h,h'}(\mathbf{s})\): spatial patterns of how shape features are associated with connectivity.
- **Posterior inclusion probabilities (PIPs)**: the selection probabilities for each subnetwork-level vertex-edge pair, quantifying confidence in shape-FC associations.
<!-- - **Credible bands / intervals** for \(\tau_e(\mathbf{s})\). -->
- **Subregion-specific connectivity effects**: identifying which subregional shape features are linked to functional connectivity.
- **Edge ranking / selection summary**: prioritizing which connectivity edges are robustly associated with shape.
- **Visualization-ready maps**: overlay coefficient estimates on surface meshes.

---

### Dependencies & Environment

- **Matlab** (tested with R2019b – R2024a)
- Basic MATLAB toolboxes (Statistics, Linear Algebra)
- No external packages required; `trandn.m` is included.

<!-- To run the example simulation, ensure paths are properly set (e.g. add project folder to MATLAB path). -->

---


📌 Citation and licensing information can be added once the paper is published / accepted.

<!-- ## Citation -->

<!-- This project is released under the **MIT License**. See `LICENSE` for details. -->

<!-- If you use COSR in your work, please cite: -->

<!-- > Ding, S., et al. (2025). *A Novel Bayesian Framework Uncovering Brain Connectivity-to-Shape Relationship in Preclinical Alzheimer’s Disease*. *Annals of Applied Statistics* (submitted / in revision). -->

<!-- You may also include a DOI or arXiv reference once available. -->



