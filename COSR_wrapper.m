function varargout = COSR_wrapper(error_model, A, M, Sv, x, varargin)
% COSR_wrapper  Dispatch helper for Connectivity-on-Shape Regression (COSR) implementations (COSR package)
%
% USAGE
%   result = COSR_wrapper(error_model, A, M, Sv, x, <params>)
%   or
%   result = COSR_wrapper(error_model, A, M, [], x, params)  % when params contains precomputed Psi & Lambda_sqrt
%
% DESCRIPTION
%   Lightweight dispatcher that forwards provided dataset and initialization
%   arguments to one of the COSR implementations. The wrapper accepts two
%   common calling patterns:
%     1) Provide spatial coordinates `Sv` (S-by-d) so the wrapper can compute
%        basis functions (Psi, Lambda_sqrt) automatically and then run the
%        chosen inference routine.
%     2) Provide a `params` struct that already contains precomputed
%        `Psi` and `Lambda_sqrt` (and other hyperparameters). In this case
%        `Sv` can be passed as `[]` and is optional.

%   Use the params struct mode (recommended) for reproducible examples; when
%   a params struct is supplied the wrapper auto-fills sensible defaults so
%   smoke tests run out-of-the-box.
%
% DATA CONTRACT (required dataset inputs)
%   A           : (V, V, n)   symmetric connectivity matrices (zero diag)
%   M           : (n, S)      shape measures at S spatial locations
%   Sv          : (S, d)      spatial coordinates for the S shape locations, d=2 or 3
%                 NOTE: `Sv` is required only when you do NOT provide
%                 precomputed basis functions. If you pass a `params` struct
%                 that contains both `Psi` and `Lambda_sqrt`, `Sv` may be
%                 omitted (pass `[]`).
%   x           : (n, p)      subject-level covariates (may be empty)
%
% CONTROL / OUTPUT
%   error_model  : 'MCMC_BF' | 'MCMC_IND' | 'VB_IND'
%   Inputs may be passed as a params struct (recommended) or as strict
%   positional arguments matching the called implementation. The wrapper
%   returns a single struct with at least:
%     .estimates : convenience point estimates (alpha, tau, Z, Gamma, B, H_new...)
%     .samples   : MCMC retained samples (when applicable)
%     .fitted_errors : diagnostic output from `simu_diagnosis_fitted` (when available)
%
% QUICK EXAMPLE (runnable smoke test)
%   The example below is adapted from `example_simu/COSR_wrapper_example_with_simdata.m`.
%   It assumes the repository helper functions (simu_data_gen_2d, simu_diagnosis_*, etc.) are
%   available on MATLAB's path or in `COSR/example_simu/`.
%
%   % Setup (quick smoke)
%   seed = 2025; rng(seed);
%   V = 20; n = 40; s = 6; S = s*s; p = 2; H = 3;
%   sigma2_e = 0.5; sigma2_gamma = 1; delta = 0.5;
%   [A, M, x, Psi, Lambda_sqrt, B_true, BM, Gamma_x, tau_true, Z_true, ...
%       w_true, gamma1_true, gamma2_true, sigma2_e_actual, SNR_A, SNR_BM, Sv] = ...
%       simu_data_gen_2d(n, V, H, p, s, sigma2_e, sigma2_gamma, delta, 1, seed);
%
%   % Example A: provide Sv and let wrapper compute Psi/Lambda_sqrt
%   params = struct(); params.seed = seed; params.H = H; params.nsamples = 200; params.burnin = 100; params.delta = delta;
%   result = COSR_wrapper('MCMC_IND', A, M, Sv, x, params);
%
%   % Example B: precompute Psi/Lambda_sqrt and pass via params (Sv optional)
%   params = struct(); params.seed = seed; params.H = H; params.nsamples = 200; params.burnin = 100; params.delta = delta; Lest = min(10, size(Psi,1)); params.Psi = Psi(1:Lest,:); params.Lambda_sqrt = Lambda_sqrt(1:Lest);
%   result = COSR_wrapper('MCMC_IND', A, M, [], x, params);
%
%   % Inspect estimates
%   est = result.estimates; disp(est.H_new);
%
% TIPS / TROUBLESHOOTING
% - Keep `nsamples` small for smoke tests (e.g., 200) and increase for production.
% - If MATLAB complains about missing helpers (trandn, munkres, randindex, simu_data_gen_2d),
%   ensure `COSR/example_simu` (or the folder containing those files) is on the MATLAB path.
% - If you see 'Output argument ... not assigned', open the helper file and verify it is a
%   valid MATLAB function (no markdown fences or extraneous text).
%
% SEE ALSO
%   COSR_VB_IND, COSR_MCMC_IND, COSR_MCMC_BF, example_simu/COSR_wrapper_example_with_simdata.m
%
% Notes: the underlying implementations contain the algorithmic logic. This wrapper
% documents the input contract and offers convenience defaults for quick reproducible
% examples.

% AUTHORS & CONTACT
%   Author: Shengxian Ding <naomidsx@gmail.com>
%   Date:   2025-09-29
%   Repository: Shape_Connectivity_Integration / COSR
%   Contact: For issues or questions, file a GitHub issue or email the author above.

if nargin < 1
    error('COSR_wrapper:MissingArg', 'error_model must be provided');
end

% Flexible input validation:
% Require A, M, x always. Require either Sv (spatial coordinates) OR a
% params struct containing both Psi and Lambda_sqrt so the wrapper can
% compute basis functions when needed.
missingInputs = {};
% Always required
if isempty(A), missingInputs{end+1} = 'A'; end
if isempty(M), missingInputs{end+1} = 'M'; end
if isempty(x), missingInputs{end+1} = 'x'; end

% Check whether caller provided a single params struct with Psi/Lambda_sqrt
hasParamsStruct = (numel(varargin) == 1) && isstruct(varargin{1});
hasPsiLambda = hasParamsStruct && isfield(varargin{1}, 'Psi') && isfield(varargin{1}, 'Lambda_sqrt');

% Sv is required only when Psi & Lambda_sqrt are NOT supplied via params
if isempty(Sv) && ~hasPsiLambda
    missingInputs{end+1} = 'Sv';
end

if ~isempty(missingInputs)
    if hasPsiLambda
        error('COSR_wrapper:MissingArg', 'Missing required dataset inputs: %s. Note: Psi and Lambda_sqrt provided in params, so Sv is optional.', strjoin(missingInputs, ', '));
    else
        error('COSR_wrapper:MissingArg', 'Missing required dataset inputs: %s. Provide either Sv (spatial coordinates) or pass a params struct with fields ''Psi'' and ''Lambda_sqrt''.', strjoin(missingInputs, ', '));
    end
end

% get sizes
[V, ~, n] = size(A);
V2 = V * (V - 1) / 2;
p = size(x, 2);


valid_models = {'MCMC_BF', 'MCMC_IND', 'VB_IND'};
modelKey = upper(error_model);
if ~ismember(modelKey, valid_models)
    error('COSR_wrapper:BadModel', 'Unsupported error_model: %s', error_model);
end


% Define required field orders for each supported model. These lists
% mirror the positional argument order of the underlying functions.
model_fields.MCMC_BF = {'Psi','Lambda_sqrt',...
    'beta','alpha','Z','w','gamma1','gamma2',...
    'Phi','sigma2_tau','sigma2_alpha','a_tau','a_alpha','b2_tau','b2_alpha','delta',...
    'g','sigma2_gamma','nsamples','burnin','thinning','seed','sigma2_Lambda','Kappa','Lambda',...
    'sigma2_f_vec','a_f','b_f'};
model_fields.MCMC_IND = {'Psi','Lambda_sqrt',...
    'alpha','Z','w','gamma1','gamma2','sigma2_e','sigma2_tau',...
    'sigma2_alpha','a_e','a_tau','a_alpha','b2_e','b2_tau','b2_alpha','delta','g','sigma2_gamma',...
    'nsamples','burnin','thinning','seed'};
model_fields.VB_IND = {'Psi','Lambda_sqrt',...
    'beta','alpha','pi_alpha_neg','pi_alpha_zero','pi_alpha_pos',...
    'E_sigma2_e_inv','E_sigma2_tau_inv','E_sigma2_alpha_inv','Z','w',...
    'gamma1','sigma2_gamma1','gamma2','Sigma_gamma2',...
    'E_a_e_inv','E_a_tau_inv','E_a_alpha_inv','delta','g','sigma2_gamma',...
    'nsamples','tol'};

fields = model_fields.(modelKey);



% If a single struct argument is passed after error_model, expand it using
% a per-model ordered field list so that parameters are forwarded in the
% correct order. If positional args are passed, forward them unchanged.
params = varargin{1};

% Default hyper-parameters
if ~isfield(params, 'seed'), params.seed = 202412; end
rng(params.seed);

% Choose small default H and K
H_default = min(3, V);
if ~isfield(params, 'H'), params.H = H_default; end
H = params.H;
H2 = H * (H + 1) / 2;
K_default = min(2, max(1, p));

% calulate eigenfunctions and eigenvalues if not provided
%   Psi         : (L, S)      basis functions (L x S)
%   Lambda_sqrt : (L, 1)      sqrt eigenvalues for the basis
if ~isfield(params, 'Psi') || ~isfield(params, 'Lambda_sqrt')
    % Compute the RBF kernel matrix
    if ~isfield(params, 'phi'), params.phi = 2; end
    if ~isfield(params, 'nu'), params.nu = 2; end
    D = pdist2(Sv, Sv); % Pairwise squared distances
    K = exp(- params.phi * D.^params.nu); % RBF kernel matrix
    % Compute eigenvalues and eigenvectors
    [eigvecs, eigvals_matrix] = eig(K); % Eigen decomposition
    eigvals = diag(eigvals_matrix); % Extract eigenvalues
    [~, idx] = sort(eigvals, 'descend'); % Sort eigenvalues and eigenvectors in descending order
    eigvals = eigvals(idx);
    eigvecs = eigvecs(:, idx);
    eigvecs = eigvecs ./ vecnorm(eigvecs); % Normalize eigenvectors
    eigen_proportion = 0.95; % Proportion of variance to capture
    [~ , L]  = max(cumsum(eigvals)/sum(eigvals) > eigen_proportion);
    if ~isfield(params, 'L'), params.L = min(20, L); end
    params.Psi = eigvecs(:, 1:params.L)'; % (L, S)
    params.Lambda_sqrt = sqrt(eigvals(1:params.L)); % (L, 1)
end
[L, S] = size(params.Psi);
disp(['Using L = ', num2str(L), ' basis functions to represent shape data.']);

if ~isfield(params, 'beta'), params.beta = randn(H2, L); end
if ~isfield(params, 'alpha'), params.alpha = params.beta * (params.Lambda_sqrt .*  params.Psi); end % (H2,S)
% if ~isfield(params, 'alpha'), params.alpha = randn(H2, S) * 0.1; end
if ~isfield(params, 'w'), params.w =  rand(1, H); params.w = params.w / sum(params.w); end
if ~isfield(params, 'Z')
    params.Z = mnrnd(1, params.w, V);
end
if ~isfield(params, 'gamma1'), params.gamma1 = randn(V, K_default); end
if ~isfield(params, 'gamma2'), params.gamma2 = randn(p, K_default); end
if ~isfield(params, 'sigma2_e'), params.sigma2_e = rand; end

if ~isfield(params, 'sigma2_tau'), params.sigma2_tau = rand; end
if ~isfield(params, 'sigma2_alpha'), params.sigma2_alpha = 0.1; end
if ~isfield(params, 'delta')
    params.delta = 0.5;
else
    disp(['Using provided delta: ', num2str(params.delta)]);
end

if ~isfield(params, 'nsamples'), params.nsamples = 1000; end
if ~isfield(params, 'burnin'), params.burnin = 500; end
if ~isfield(params, 'thinning'), params.thinning = 1; end

% hyperparameters
params.g = ones(1, H);
params.sigma2_gamma = 10;
params.a_e = 1;
params.a_tau = 1;
params.a_alpha = 1;
params.b2_e = 1;
params.b2_tau = 1;
params.b2_alpha = 1;


% Map provided *_0 initials (if any) into model-specific names.
switch modelKey
    
    case 'VB_IND'
        % Map simple initials to VB names
        pi_alpha_all = rand(H2, S, 3);
        pi_alpha_all = pi_alpha_all ./ sum(pi_alpha_all, 3);
        params.pi_alpha_neg = pi_alpha_all(:, :, 1);
        params.pi_alpha_zero = pi_alpha_all(:, :, 2);
        params.pi_alpha_pos = pi_alpha_all(:, :, 3);
        
        % sensible defaults for gamma / variational params
        if ~isfield(params, 'sigma2_gamma1'), params.sigma2_gamma1 = rand(V, K_default); end
        if ~isfield(params, 'Sigma_gamma2'), params.Sigma_gamma2 = repmat(eye(p), [1, 1, K_default]); end
        
        % expectations and auxiliary parameters
        params.E_sigma2_alpha_inv = 1 / params.sigma2_alpha;
        params.E_sigma2_e_inv = 1 / params.sigma2_e;
        params.E_sigma2_tau_inv = 1 / params.sigma2_tau;
        
        params.E_a_e_inv = 1/2;
        params.E_a_tau_inv = 1/2;
        params.E_a_alpha_inv = 1/2;
        params.tol = 1e-3;
        
    case 'MCMC_BF'
        if ~isfield(params, 'Phi'), params.Phi = eye(V2); end
        D = 5;
        params.Kappa = randn(n, D);
        params.Lambda = randn(V2, D);
        params.sigma2_f_vec = rand(1, V2);
        params.a_f = 0.01;
        params.b_f = 0.01;
end


% Construct args in the exact expected order (all fields now exist)
args = cell(1, numel(fields));
for i = 1:numel(fields)
    if ~isfield(params, fields{i})
        error('COSR_wrapper:MissingField', 'Missing required field for model %s: %s', modelKey, fields{i});
    end
    args{i} = params.(fields{i});
end


% Store results in a structured format
result = struct();

% Call the appropriate COSR implementation
switch modelKey
    case 'MCMC_IND'
        [~, alpha_s, sigma2_e_s, sigma2_tau_s, sigma2_alpha_s, ...
            Z_s, ~, gamma1_s, gamma2_s, tau_tilde_s, H_new] = ...
            COSR_MCMC_IND(A, M, x, args{:});
        result.samples = struct('alpha', alpha_s, 'sigma2_e', sigma2_e_s, 'sigma2_tau', sigma2_tau_s, 'sigma2_alpha', sigma2_alpha_s, 'Z', Z_s, 'tau_tilde', tau_tilde_s);
        
    case 'VB_IND'
        [~, alpha_hat, tau_hat, ~, ~, ~, ...
            E_Z, ~, ~, ~, ~, Gamma_hat, H_new, ~, ~, ~] = ...
            COSR_VB_IND(A, M, x, args{:});
        
    case 'MCMC_BF'
        [~, alpha_s, sigma2_tau_s, sigma2_alpha_s, ...
            Z_s, ~, gamma1_s, gamma2_s, tau_tilde_s, Phi_hat, H_new] = ...
            COSR_MCMC_BF(A, M, x, args{:});
        
        result.samples = struct('alpha', alpha_s, 'Phi_e', Phi_hat, 'sigma2_tau', sigma2_tau_s, 'sigma2_alpha', sigma2_alpha_s, 'Z', Z_s, 'tau_tilde', tau_tilde_s);
end

% Post-processing common to all models
H2_new = H_new * (H_new + 1) / 2;
disp(['Inferred number of clusters H_new: ', num2str(H_new)]);

if startsWith(modelKey, 'MCMC')
    % Compute estimators (means over retained samples)
    if exist('tau_tilde_s', 'var') && size(tau_tilde_s, 1) ~= H2
        disp('Permuting tau_tilde_s dimensions');
        tau_tilde_s = permute(tau_tilde_s, [2, 1, 3]);
    end
    alpha_thresholded_all = abs(alpha_s(1:H2_new, :, :)) > params.delta; % (H2, S, nsamples)
    alpha_hat = mean(alpha_thresholded_all, 3); % (H2_new, S)
    alpha_hat_thresholded = alpha_hat > 0.5; % (H2_new, S)
    
    tau_all = tau_tilde_s(1:H2_new, :, :) .* alpha_thresholded_all;
    tau_hat = mean(tau_all, 3); % (H2_new, S)
    
    Z_all_probs = mean(Z_s(:, 1:H_new, :), 3); % (V, H)
    [~, Z_hat_idx] = max(Z_all_probs, [], 2);
    
    gamma1_s_reshape = permute(gamma1_s, [1, 4, 3, 2]); % (V, 1, nsamples, K)
    Gamma_all = sum(permute(pagemtimes(gamma1_s_reshape, "none", gamma1_s_reshape, "transpose"), ...
        [1, 2, 5, 3, 4]) .* permute(gamma2_s, [4, 5, 1, 3, 2]), 5); % (V, V, p, nsamples)
    Gamma_hat = mean(Gamma_all, 4); % (V, V, p)
    
elseif startsWith(modelKey, 'VB')
    [~, Z_hat_idx] = max(E_Z, [], 2); % (V, 1)
    tau_hat = tau_hat(1:H2_new, :);
    alpha_hat = alpha_hat(1:H2_new, :);
    alpha_hat_thresholded = abs(alpha_hat) > params.delta; % (H2_new, S)
end

Z_hat = double(bsxfun(@eq, (1:H_new), Z_hat_idx)); % (V, H_new)

T_hat = zeros(S, H_new, H_new);
mask_h_tri = triu(true(H_new, H_new));
mask_h_triu = triu(true(H_new, H_new), 1);
mask_h_tril = tril(true(H_new, H_new), -1);
T_hat(:, mask_h_tri) = tau_hat';
T_hat(:, mask_h_tril) = T_hat(:, mask_h_triu);
B_hat = pagemtimes(pagemtimes(Z_hat, permute(T_hat, [2, 3, 1])), Z_hat'); % (V, V, S)

[fitted_errors] = simu_diagnosis_fitted(A, M, x, [], B_hat, Gamma_hat);
% disp(fitted_errors);


% Store final results
result.estimates = struct('alpha', alpha_hat, 'alpha_thresholded', alpha_hat_thresholded, ...
    'tau', tau_hat, 'Z', Z_hat, 'Gamma', Gamma_hat, 'B', B_hat, 'H_new', H_new);
result.fitted_errors = fitted_errors;

varargout{1} = result;

end
