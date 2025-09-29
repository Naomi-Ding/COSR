%% MCMC update: single iteration of sample draw

function [beta_samples, alpha_samples, sigma2_e_samples, sigma2_tau_samples, ...
    sigma2_alpha_samples, Z_samples, w_samples, gamma1_samples, gamma2_samples, ...
    tau_tilde_samples, H] = COSR_MCMC_IND(A, M, x, Psi, Lambda_sqrt, ...
    alpha, Z, w, gamma1, gamma2, sigma2_e, sigma2_tau, sigma2_alpha, ...
    a_e, a_tau, a_alpha, b2_e, b2_tau, b2_alpha, delta, g, sigma2_gamma, ...
    nsamples, burnin, thinning, seed)
%%% Data (shapes):
%%%  A            (V, V, n)   - symmetric connectivity matrices (diagonal 0)
%%%  M            (n, S)      - shape measures evaluated at S spatial locations
%%%  x            (n, p)      - confounders / covariates
%%%  Psi          (L, S)      - basis/eigenfunctions (L x S)
%%%  Lambda_sqrt  (L, 1)      - sqrt eigenvalues (L x 1)

%%% Initial values and hyperparameters (shapes):
%%%  alpha        (H2, S)  - latent indicator for tau_tilde (upper-triangular index)
%%%  Z            (V, H)   - cluster memberships (one-hot rows)
%%%  w            (1, H)   - cluster weights
%%%  gamma1       (V, K)   - node-specific factors for regression on x
%%%  gamma2       (p, K)   - covariate factors for regression on x
%%%  sigma2_*     scalars  - variance hyperparameters
%%%  a_*, b_*     scalars  - variance hyperparameters for priors
%%%  delta        scalar   - threshold for alpha -> tau_tilde
%%%  g            scalar   - concentration parameter for cluster prior

%%% Chain settings:
%%%  nsamples, burnin, thinning, seed

%%% Outputs: samples of beta, alpha, sigma2_tau, sigma2_alpha, Z, w, gamma1, gamma2
%%%  beta_samples       (L, H2, nsamples)
%%%  alpha_samples     (H2, S, nsamples)
%%%  sigma2_tau_samples(nsamples, 1)
%%%  sigma2_alpha_samples(nsamples, 1)
%%%  Z_samples         (V, H, nsamples)
%%%  w_samples         (H, nsamples)
%%%  gamma1_samples    (V, K, nsamples)
%%%  gamma2_samples    (p, K, nsamples)
%%%  tau_tilde_samples (H2, S, nsamples) - calculated as Psi' * (alpha .* Lambda_sqrt)
%%%  Phi               (V2, V2) - final estimate of residual precision matrix
%%%  H                scalar   - number of clusters

%%% Notes:
%%%  - H2 = H*(H+1)/2 denotes the number of unique upper-triangular (h,h') pairs.
%%%  - Psi_lambda = Psi' .* Lambda_sqrt' is used internally for basis transforms.

if isempty(seed)
    seed = 202412;
end
rng(seed);

[V, H] = size(Z);
[L, S] = size(Psi);
H2 = H * (H + 1) / 2;
[n, p] = size(x);
K = size(gamma1, 2); % rank K of tensor decomposition


%% saved quantities
beta_samples = zeros(H2, L, nsamples);
% tau_tilde_samples = zeros(S, H2, nsamples);
alpha_samples = zeros(H2, S, nsamples);
sigma2_e_samples = zeros(nsamples, 1);
sigma2_tau_samples = zeros(nsamples, 1);
sigma2_alpha_samples = zeros(nsamples, 1);
% a_e_samples = zeros(nsamples, 1);
% a_tau_samples = zeros(nsamples, 1);
% a_alpha_samples = zeros(nsamples, 1);
Z_samples = zeros(V, H, nsamples);
w_samples = zeros(H, nsamples);
gamma1_samples = zeros(V, K, nsamples);
gamma2_samples = zeros(p, K, nsamples);


%% pre-calculate some quantities
Psi_lambda = Psi' .* Lambda_sqrt'; % (S, L)
Psi_lambda_square = Psi_lambda' * Psi_lambda; % (L, L)
mask_h_tri = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
mask_h_tril = tril(true(H, H), -1);
mask_h_triu = triu(true(H, H), 1);
mask_h_diag = repmat(diag(true(H, 1)), 1, 1, n); % (H, H, n), diagonal element
mask_v_triu_Hn = repmat(triu(true(V, V), 1), 1, 1, H, n); % (V, V, H, n), upper tri element
mask_v_triu_n = repmat(triu(true(V, V), 1), 1, 1, n); % (V, V, n)
mask_v_triu_K = repmat(triu(true(V, V), 1), 1, 1, 1, K); % (V, V, 1, K)

sigma2_e_shape = 1/2 + n * V * (V - 1) / 4;
sigma2_tau_shape = 1/2 + L * H * (H + 1) / 4;
sigma2_alpha_shape = 1/2 + S * H * (H + 1) / 4;

[~, clusters] = max(Z, [], 2);
H_old = H;
cluster_count = 0;

%% iterations
tic;
for nn = 1:nsamples

    %% (1) calculate \sum_{v<v'}ZvZv' & \sum_{v<v'}ZvZv'Avv'
    gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
    gamma2_x_reshape = permute(x * gamma2, [3, 4, 1, 2]); % (1, 1, n, K)
    Gamma_x = sum(permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]) .* gamma2_x_reshape, 4); % (V, V, n)
    A_tilde = A - Gamma_x; % (V, V, n)
    alpha_diag = permute(abs(alpha) > delta, [3, 2, 1]); % (1, S, H2)
    M_alpha = M .* alpha_diag; % (n, S, H2)
    M_alpha_Psi_lambda = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
    M_alpha_Psi_lambda_square = pagemtimes(M_alpha_Psi_lambda, 'transpose',...
        M_alpha_Psi_lambda, 'none'); % (L, L, H2)

    Zh = sum(Z, 1); % (1, H)
    Zh2 = Zh.' * Zh; % (H, H)
    Zh_diag = Zh .* (Zh - 1) / 2; % (1, H)
    ZvZv_sum = diag(Zh_diag) + triu(Zh2, 1); % (H, H)
    ZvZv_sum_triu = reshape(ZvZv_sum(mask_h_tri), H2, 1); % (H2, 1)

    ZAZ = pagemtimes(pagemtimes(Z.', A_tilde), Z); % (H, H, n)
    ZAZ_diag = permute(Z, [1, 3, 2]) .* permute(Z, [3, 1, 2]) .* permute(A_tilde, [1, 2, 4, 3]); % (V, V, H, n)
    ZAZ(mask_h_diag) = sum(reshape(ZAZ_diag(mask_v_triu_Hn), [], H, n), 1);
    ZAZ_triu = reshape(ZAZ(repmat(mask_h_tri, 1, 1, n)), H2, n); % (H2, n)

    %% (2) sample beta
    % mean and covariance for beta
    Sigma_beta_inv = reshape(ZvZv_sum_triu, 1, 1, H2) .* M_alpha_Psi_lambda_square / sigma2_e ...
        + repmat(eye(L, L), 1, 1, H2) / sigma2_tau + repmat(Psi_lambda_square, 1, 1, H2) / sigma2_alpha; % (L, L, H2)
    Sigma_beta = pageinv(Sigma_beta_inv); % (L, L, H2)
    mu_beta1 = pagemtimes(permute(ZAZ_triu, [3, 2, 1]), M_alpha_Psi_lambda); % (1, L, H2)
    mu_beta2 = permute(mu_beta1, [2, 1, 3]) / sigma2_e + ...
        permute(alpha * Psi_lambda / sigma2_alpha, [2, 3, 1]); % (L, 1, H2)
    mu_beta = permute(pagemtimes(Sigma_beta, mu_beta2), [3, 1, 2]); % (H2, L , 1)
    % sample from multivariate normal
    beta = mvnrnd(mu_beta, Sigma_beta); % (H2, L)
    beta_samples(1:H2, :, nn) = beta;

    %% (3) sample alpha(s_j), j = 1:S
    tau_tilde = permute(beta * Psi_lambda.', [3, 2, 1]); % (1, S, H2)
    tau_tilde_M_sj = M .* tau_tilde; % (n, S, H2)
    c1 = pagemtimes(permute(ZAZ_triu, [3, 2, 1]), tau_tilde_M_sj); % (1, S, H2)
    tau_tilde_bound1 = (- delta - tau_tilde) / sqrt(sigma2_alpha); % (1, S, H2)
    tau_tilde_bound2 = (delta - tau_tilde) / sqrt(sigma2_alpha); % (1, S, H2)
    pi_neg = normcdf(tau_tilde_bound1); % (1, S, H2)
    pi_zero = normcdf(tau_tilde_bound2) - normcdf(tau_tilde_bound1); % (1, S, H2)
    pi_pos = (1 - normcdf(tau_tilde_bound2)); % (1, S, H2)
    % sample from each truncated normal
    TN_neg = reshape(trandn(repelem(-Inf, S * H2), tau_tilde_bound1), [1, S, H2]); % (1, S, H2)
    TN_zero = reshape(trandn(tau_tilde_bound1, tau_tilde_bound2), [1, S, H2]); % (1, S, H2)
    TN_pos = reshape(trandn(tau_tilde_bound2, repelem(Inf, S * H2)), [1, S, H2]); % (1, S, H2)
    TN_all = cat(1, TN_neg, TN_zero, TN_pos) * sqrt(sigma2_alpha) + tau_tilde; % (3, S, H2)
    ZvZv_sum_triu_reshape = reshape(ZvZv_sum_triu, 1, 1, H2); % (1, 1, H2)
    for j = 1:S
        tau_M_sj = tau_tilde_M_sj .* alpha_diag; % (n, S, H2)
        tau_M = sum(tau_M_sj, 2); % (n, 1, H2)
        tau_M_non_sj = tau_M - tau_M_sj; % (n, S, H2)
        c2 = sum((tau_M_non_sj(:, j, :) + tau_tilde_M_sj(:, j, :) / 2) .* ...
            tau_tilde_M_sj(:, j, :), 1) .* ZvZv_sum_triu_reshape; % (1, 1, H2)
        c = exp((c1(:, j, :) - c2) / sigma2_e); % (1, 1, H2)
        pi_neg_c = pi_neg(:, j, :) .* c; % (1, 1, H2)
        pi_pos_c = pi_pos(:, j, :) .* c; % (1, 1, H2)
        pi_all = cat(1, pi_neg_c, pi_zero(:, j, :), pi_pos_c); % (3, 1, H2)
        pi_all_norm = pi_all ./ sum(pi_all, 1); % (3, 1, H2)
        % decide which component to sample from
        components_prob = rand(1, H2);
        cumsum_pi = cumsum(reshape(pi_all_norm, 3, H2), 1); % (3, H2)
        [~, components_idx] = max(components_prob < cumsum_pi, [], 1); % (1, H2)
        components_idx_linear = sub2ind([3, H2], components_idx, 1:H2); % (1, H2)
        TN_all_sj = reshape(TN_all(:, j, :), 3, H2); % (3, H2)
        alpha(:, j) = TN_all_sj(components_idx_linear); % (1, H2)
        alpha_diag(1, j, :) = abs(alpha(:, j)) > delta; % (1, S, H2)
        % M_alpha = M .* alpha_diag; % (n, S, H2)
    end
    alpha_samples(1:H2, :, nn) = alpha;

    %% sample sigma2_e
    % update quantities related to alpha
    % tau_M = pagemtimes(M_alpha, permute(tau_tilde, [2, 1, 3])); % (n, 1, H2)
    tau_M = sum(tau_tilde_M_sj .* alpha_diag, 2); % (n, 1, H2)
    % reshape tau_M to H x H
    tau_M_HtimesH = zeros(n, H, H);
    tau_M_HtimesH(:, mask_h_tri) = tau_M; % add upper tri & diag
    tau_M_HtimesH(:, mask_h_tril) = tau_M_HtimesH(:, mask_h_triu); % add lower tri
    m_tilde = pagemtimes(pagemtimes(Z, permute(tau_M_HtimesH, [2, 3, 1])), Z'); % (V, V, n)
    residual = permute(A_tilde - m_tilde, [3, 4, 1, 2]); % (n, 1, V, V)
    residual_sum_square = pagemtimes(residual, 'transpose', residual, 'none'); % (1, 1, V, V)
    residual_sum_square_triu_sum = sum(triu(reshape(residual_sum_square, V, V), 1), [1, 2]); % scalar
    sigma2_e_rate = residual_sum_square_triu_sum / 2 + 1 / a_e;
    sigma2_e = 1 / gamrnd(sigma2_e_shape, 1 / sigma2_e_rate); % scalar
    sigma2_e_samples(nn) = sigma2_e;

    %% sample sigma2_tau
    sigma2_tau_rate = sum(beta .* beta , [1, 2]) / 2 + 1 / a_tau;
    sigma2_tau = 1 / gamrnd(sigma2_tau_shape, 1 / sigma2_tau_rate); % scalar
    sigma2_tau_samples(nn) = sigma2_tau;

    %% sample sigma2_alpha
    sigma2_alpha_rate = sum((alpha' - reshape(tau_tilde, S, H2)).^2, "all") / 2 + 1 / a_alpha;
    sigma2_alpha = 1 / gamrnd(sigma2_alpha_shape, 1 / sigma2_alpha_rate); % scalar
    sigma2_alpha_samples(nn) = sigma2_alpha;

    % %% sample a_e, a_tau, a_alpha
    % a_e = 1 / gamrnd(1, 1 / (sigma2_e^(-1) + b2_e^(-1)));
    % a_tau = 1 / gamrnd(1, 1 / (sigma2_tau^(-1) + b2_tau^(-1)));
    % a_alpha = 1 / gamrnd(1, 1 / (sigma2_alpha^(-1) + b2_alpha^(-1)));

    %% sample Zv, v = 1, ..., V
    tau_M_HtimesH_reshape = reshape(tau_M_HtimesH, [n, 1, H, H]); % (n, 1, H, H)
    tau_M_square = reshape(pagemtimes(tau_M_HtimesH_reshape, 'transpose',...
        tau_M_HtimesH_reshape, 'none'), H, H); % (1, 1, H, H) -> (H, H)
    A_tau_M = pagemtimes(permute(A_tilde, [2, 3, 4, 1]), tau_M_HtimesH); % (V, H, H, V)
    for v = 1:V
        Z(v, :) = 0;
        tau_M_square_Z = tau_M_square * Z'; % (H, V)
        % tau_M_square_Z(:, v) = []; % remove the v-th node
        zvh_check_1 = sum(tau_M_square_Z, 2);  % (H, 1), sum over v' \neq v, for v'=1-V
        A_tau_M_Z = A_tau_M(:, :, :, v) .* Z; % (V, H, H)
        A_tau_M_Z_sum_H = sum(A_tau_M_Z, 2); % (V, 1, H)
        % A_tau_M_Z_sum_H(v, :, :) = []; % remove the v-th node, (V-1, 1, H)
        zvh_check_2 = -2 * reshape(sum(A_tau_M_Z_sum_H, 1), H, 1); % (H, 1)
        zvh_check = zvh_check_1 + zvh_check_2; % (H, 1)
        log_pi_vh = log(w') - zvh_check / (2 * sigma2_e); % (H, 1)
        log_pi_vh_max = max(log_pi_vh); % scalar
        log_pi_vh_shift = log_pi_vh - log_pi_vh_max; % (H, 1)
        pi_z_shift = exp(log_pi_vh_shift); % (H, 1)
        pi_z_shift_norm = pi_z_shift / sum(pi_z_shift); % (H, 1)
        Z(v, find(rand < cumsum(pi_z_shift_norm), 1)) = 1; % (H, 1)
    end
    Z_samples(:, 1:H, nn) = Z;

    %% check whether to shrink the clusters
    clusters_old = clusters; % (V, 1)
    [~, clusters] = max(Z, [], 2);
    clusters_diff = sum(abs(clusters - clusters_old));
    if clusters_diff == 0 % all the node memberships are the same with the previous iteration
        cluster_count = cluster_count + 1;
    else
        cluster_count = 0; % restart the counting if the node memberships are different
    end
    if (cluster_count >= 3)
        % [~, clusters] = max(E_Z, [], 2);
        % binary_clusters = bsxfun(@eq, (1:H), clusters); % (V, H)
        % cluster_prop = mean(binary_clusters, 1); % (1, H)
        zero_cluster = sum(Z, 1) == 0; % (1, H)
        H0 = sum(zero_cluster);
        if H0 > 0
            H = H_old - H0;
            disp(['Z converged at iter ', num2str(nn)]);
            disp(['new cluster number = ', num2str(H)]);
            H_old = H;
            H2 = H * (H + 1) / 2;
            % update index & other parameters
            tmp = mask_h_tri; % (H, H), upper tri
            tmp(zero_cluster, :) = 0;
            tmp(:, zero_cluster) = 0;
            H2_idx = tmp(mask_h_tri);

            % beta = beta(H2_idx, :);
            alpha = alpha(H2_idx, :);
            Z = Z(:, ~zero_cluster); % (V, H)
            g = g(~zero_cluster); % (1, H)
            tau_M_HtimesH = tau_M_HtimesH(:, ~zero_cluster, ~zero_cluster); % (n, H, H)
            % update the pre-calculated quantities
            mask_h_tri = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
            mask_h_tril = tril(true(H, H), -1);
            mask_h_triu = triu(true(H, H), 1);
            mask_h_diag = repmat(diag(true(H, 1)), 1, 1, n); % (H, H, n), diagonal element
            mask_v_triu_Hn = repmat(triu(true(V, V), 1), 1, 1, H, n); % (V, V, H, n), upper tri element
            % mask_h_tri_V2 = repmat(triu(true(H)), 1, 1, V2); % (H, H, V2)
            % mask_h_tri_n = repmat(mask_h_tri, 1, 1, n); % (H, H, n), upper triangle element
        end

    end

    %% sample w
    w_concentration = sum(Z, 1) + g; % (1, H)
    w_gamma = gamrnd(w_concentration, 1); % (1, H)
    w = w_gamma / sum(w_gamma); % (1, H)
    w_samples(1:H, nn) = w;

    %% sample gamam_1vk, v=1,...,V, k=1,...,K
    gamma2_reshape = permute(gamma2, [3, 4, 1, 2]); % (1, 1, p, K)
    m_tilde = pagemtimes(pagemtimes(Z, permute(tau_M_HtimesH, [2, 3, 1])), Z'); % (V, V, n)
    A_hat_gamma = A - m_tilde; % (V, V, n)
    x_gamma2 = x * gamma2; % (n, K)
    x_gamma2_square_sum = diag(x_gamma2' * x_gamma2); % (K, 1)
    x_gamma2_div_sigma2e = x_gamma2 / sigma2_e; % (n, K)
    x_gamma2_square_sum_div_sigma2_e = x_gamma2_square_sum / sigma2_e; % (K, 1)
    for k = 1:K
        gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
        Gamma_K = permute(permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]) .* gamma2_reshape, [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
        Gamma_K(k, :, :, :) = 0; % remove the k-th component
        Gamma_x_non_k = permute(sum(pagemtimes(Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
        A_hat_gamma_k = A_hat_gamma - Gamma_x_non_k; % (V, V, n)
        for v = 1:V
            gamma1(v, k) = 0; % (V, K)
            gamma1k_no_v_square = gamma1(:, k)' * gamma1(:, k); % scalar
            sigma2_gamma1k_inv = x_gamma2_square_sum_div_sigma2_e(k) * gamma1k_no_v_square + 1 / sigma2_gamma; % scalar
            sigma2_gamma1k = 1 / sigma2_gamma1k_inv; % scalar
            A_hat_gamma_k_no_v = reshape(A_hat_gamma_k(v, :, :), V, n); % (V, n)
            A_hat_gamma_k_no_v(v, :) = 0; % remove the v-th node
            A_gamma1k_no_v = A_hat_gamma_k_no_v' * gamma1(:, k); % (n, 1)
            mu_gamma1k = x_gamma2_div_sigma2e(:, k).' * A_gamma1k_no_v / sigma2_gamma1k_inv; % scalar
            gamma1(v, k) = normrnd(mu_gamma1k, sqrt(sigma2_gamma1k));
        end
    end
    gamma1_samples(:, :, nn) = gamma1;

    %% sample gamma_2k, k=1,...,K
    gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
    gamma1_outer = permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]); % (V, V, 1, K)
    gamma1_check = reshape(gamma1_outer(mask_v_triu_K), [], K); % (V2, K), V2 = V*(V-1)/2
    gamma1_check_square = diag(gamma1_check' * gamma1_check); % (K, K) -> (K, 1)
    x_square_sum_div_sigma2e = x' * x / sigma2_e; % (p, p)
    for k = 1:K
        Sigma_gamma2k_inv = gamma1_check_square(k) * x_square_sum_div_sigma2e + eye(p) / sigma2_gamma; % (p, p)
        Sigma_gamma2k = inv(Sigma_gamma2k_inv); % (p, p)
        Gamma_K = permute(gamma1_outer .* permute(gamma2, [3, 4, 1, 2]), [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
        Gamma_K(k, :, :, :) = 0; % remove the k-th component
        Gamma_x_non_k = permute(sum(pagemtimes(Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
        A_hat_gamma_k = A_hat_gamma - Gamma_x_non_k; % (V, V, n)
        A_hat_gamma_k_check = reshape(A_hat_gamma_k(mask_v_triu_n), [], n); % (V2, n)
        mu_gamma2k = Sigma_gamma2k_inv \ (gamma1_check(:, k)' * A_hat_gamma_k_check * x)' / sigma2_e; % (p, 1)
        gamma2(:, k) = mvnrnd(mu_gamma2k, Sigma_gamma2k); % (p, 1)
    end
    gamma2_samples(:, :, nn) = gamma2;

    %% evaluate elapsed time
    if mod(nn, 100) == 0
        disp(nn);
        toc;
    end

end

if ~isempty(burnin) && ~isempty(thinning)
    eff_idx = (burnin + 1) : (thinning) : nsamples; % indices of effective samples
    beta_samples = beta_samples(:, :, eff_idx);
    alpha_samples = alpha_samples(:, :, eff_idx);
    sigma2_e_samples = sigma2_e_samples(eff_idx);
    sigma2_tau_samples = sigma2_tau_samples(eff_idx);
    sigma2_alpha_samples = sigma2_alpha_samples(eff_idx);
    Z_samples = Z_samples(:, :, eff_idx);
    w_samples = w_samples(:, eff_idx);
    gamma1_samples = gamma1_samples(:, :, eff_idx);
    gamma2_samples = gamma2_samples(:, :, eff_idx);
end

% if nargout > 9
tau_tilde_samples = pagemtimes(beta_samples, Psi_lambda'); % (H2, S, nsamples)
% tau_tilde_samples = pagemtimes(Psi_lambda, 'none', beta_samples, 'transpose'); % (S, H2, neffsamples)
% end

end