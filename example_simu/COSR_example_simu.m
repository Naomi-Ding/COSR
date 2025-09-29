% COSR_wrapper_example_with_simdata.m
% Build a small synthetic dataset using the repo's data generator and run the wrapper

clear;
seed = 2025;
rng(seed);

% Small settings for a quick smoke test
V = 20;   % nodes (small)
n = 40;   % subjects
s = 6;    % image side length (s*s = S locations)
S = s * s;
p = 2;
H = 3;

% generator options (use the independent-error generator here; swap to IW if needed)
sigma2_e = 0.5;
sigma2_gamma = 1;
delta = 0.5;
tau_pattern = 1;

% Call the available data generator (simu_data_gen_2d)
[A, M, x, Psi, Lambda_sqrt, B_true, BM, Gamma_x, tau_true, Z_true, w_true, gamma1_true, gamma2_true, sigma2_e_actual, SNR_A, SNR_BM] = ...
    simu_data_gen_2d(n, V, H, p, s, sigma2_e, sigma2_gamma, delta, tau_pattern, seed, false, 0.95, 20, 1);


% split data
n_train = n * 0.8;
n_val = n * 0.1;
n_test = n - n_train - n_val;

% Random permutation of indices
indices = randperm(n);
% Assign to train, validation, and test sets
train_idx = indices(1:n_train);
val_idx = indices(n_train+1:n_train+n_val);
test_idx = indices(n_train+n_val+1:end);

A_train = A(:,:,train_idx);
A_val = A(:,:,val_idx);
A_test = A(:,:,test_idx);
M_train = M(train_idx, :);
M_val = M(val_idx, :);
M_test = M(test_idx, :);
x_train = x(train_idx, :);
x_val = x(val_idx, :);
x_test = x(test_idx, :);
BM_train = BM(:,:,train_idx);
BM_val = BM(:,:,val_idx);
BM_test = BM(:,:,test_idx);


% build params for wrapper (use small chains for smoke test)
params = struct();
params.seed = 2025;
params.H = H;
params.nsamples = 200; % tiny chain for smoke
params.burnin = 100;
params.thinning = 1;
params.delta = delta;
% include dataset fields used by wrapper defaults when params struct is used
Lest = min(10, size(Psi, 1)); % number of basis used
params.Psi = Psi(1:Lest, :);
params.Lambda_sqrt = Lambda_sqrt(1:Lest);

% Run wrapper (MCMC_IND for easier comparision with tau_true)
disp('Running COSR_wrapper (MCMC_IND) on generated data...');
result = COSR_wrapper('MCMC_IND', A_train, M_train, x_train, Psi(1:Lest,:), Lambda_sqrt(1:Lest), params);

% Extract estimates
est = result.estimates;
if isfield(est, 'tau')
    tau_hat = est.tau; % (H2, S)
else
    warning('tau estimate not present in wrapper result');
    tau_hat = [];
end

if isfield(est, 'alpha_thresholded')
    alpha_thresh = est.alpha_thresholded;
else
    alpha_thresh = [];
end
% Evaluate performance using the repo's diagnostic helpers (same as main scripts)
H2 = H * (H + 1) / 2;

% Compute Z_hat_idx from estimated cluster assignment (est.Z expected (V,H_new))
if isfield(est, 'Z')
    [~, Z_hat_idx] = max(est.Z, [], 2);
else
    warning('Estimated Z not present; attempting to derive from result.samples');
    Z_hat_idx = [];
end

% Call tau diagnostic helper (mirrors main script)
if exist('tau_true','var') && ~isempty(Z_true) && isfield(est, 'alpha') && isfield(est, 'tau') && ~isempty(Z_hat_idx)
    if size(tau_true, 1) ~= H2
        tau_true = tau_true';
    end
    [alpha_hat_out, alpha_hat_thresholded_out, tau_hat_out, Z_df, df_tau_indicator, tau_errors] = ...
        simu_diagnosis_estimation_tau(tau_true, Z_true, est.H_new, est.alpha, ...
        est.alpha_thresholded, est.tau, Z_hat_idx);
    
    disp('membership diagnostics:');
    disp(Z_df);
    if ~isempty(df_tau_indicator)
        disp('tau selection summary (first 6 rows):');
        disp(df_tau_indicator(1:min(6,end), :));
    end
else
    warning('Skipping tau diagnostics; missing required variables (tau_true, Z_true, est.alpha, est.tau, or Z_hat_idx).');
    alpha_hat_out = [];
    df_tau_indicator = [];
    tau_errors = [];
end

% Call B diagnostic helper
if isfield(est, 'B') && exist('B_true','var')
    [AUC_B_nodes, df_B_vtx_indicator, df_B_indicator, B_errors] = simu_diagnosis_estimation_B(B_true, est.B);
    disp('B diagnostics:'); disp(df_B_indicator);
else
    warning('Skipping B diagnostics; missing est.B or B_true');
    df_B_indicator = [];
    B_errors = [];
end

% Compute fitted errors on each split using the repo helper
if isfield(est, 'B') && isfield(est, 'Gamma')
    fitted_errors_train = simu_diagnosis_fitted(A_train, M_train, x_train, BM_train, est.B, est.Gamma);
    disp('Fitted errors (train):'); disp(fitted_errors_train);
    fitted_errors_val = simu_diagnosis_fitted(A_val, M_val, x_val, BM_val, est.B, est.Gamma);
    disp('Fitted errors (val):'); disp(fitted_errors_val);
    fitted_errors_test = simu_diagnosis_fitted(A_test, M_test, x_test, BM_test, est.B, est.Gamma);
    disp('Fitted errors (test):'); disp(fitted_errors_test);
else
    warning('Skipping fitted error diagnostics; est.B or est.Gamma missing');
    fitted_errors_train = [];
    fitted_errors_val = [];
    fitted_errors_test = [];
end

% Save results to file for inspection
save('COSR_wrapper_example_with_simdata_result.mat', 'result', 'tau_true', 'B_true', 'Z_true', ...
    'tau_errors', 'df_tau_indicator', 'B_errors', 'df_B_indicator', 'fitted_errors_train');

disp('Done. Result + diagnostics saved to COSR_wrapper_example_with_simdata_result.mat');
