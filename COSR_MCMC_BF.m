%% MCMC update with correlated residuals using Bayesian Factor analysis: multiple iterations of sample draw

function [beta_samples, alpha_samples, sigma2_tau_samples, ...
    sigma2_alpha_samples, Z_samples, w_samples, gamma1_samples, gamma2_samples, ...
    tau_tilde_samples, Phi, H] = COSR_MCMC_BF(A, M, x, Psi, Lambda_sqrt, ...
    beta, alpha, Z, w, gamma1, gamma2, Phi, sigma2_tau, sigma2_alpha, ...
    a_tau, a_alpha, b2_tau, b2_alpha, delta, g, sigma2_gamma, ...
    nsamples, burnin, thinning, seed, sigma2_Lambda, Kappa, Lambda, sigma2_f_vec, a_f, b_f)
%%% Data (shapes):
%%%  A            (V, V, n)   - symmetric connectivity matrices (diagonal 0)
%%%  M            (n, S)      - shape measures evaluated at S spatial locations
%%%  x            (n, p)      - confounders / covariates
%%%  Psi          (L, S)      - basis/eigenfunctions (L x S)
%%%  Lambda_sqrt  (L, 1)      - sqrt eigenvalues (L x 1)

%%% Initials and priors (shapes):
%%%  beta         (L, H2)  - basis coefficients for tau_tilde (H2 = H*(H+1)/2)
%%%  alpha        (H2, S)  - latent indicators for tau_tilde
%%%  Z            (V, H)   - cluster memberships (one-hot rows)
%%%  w            (1, H)   - cluster weights
%%%  gamma1       (V, K)   - node-specific factors
%%%  gamma2       (p, K)   - covariate factors
%%%  Phi          (V2,V2)  - precision matrix for factor-model residuals (V2 = V*(V-1)/2)
%%%  sigma2_*     scalars  - variance hyperparameters
%%%  a_*, b_*     scalars  - variance hyperparameters for priors
%%%  delta        scalar   - threshold for alpha -> tau_tilde
%%%  g            scalar   - concentration parameter for cluster prior

%%% Factor model specific:
%%%  Kappa        (n, D)   - latent factors for observations
%%%  Lambda       (V2, D)  - loadings per upper-triangular residual element
%%%  sigma2_f_vec (1, V2)  - idiosyncratic variances for factor model

%%% Chain settings: nsamples, burnin, thinning, seed

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
%%%  - H2 = H*(H+1)/2 denotes number of unique (h,h') pairs.
%%%  - Psi_lambda = Psi' .* Lambda_sqrt' used for basis transforms.

if isempty(seed)
    seed = 202412;
end
rng(seed);

[V, H] = size(Z);
[L, S] = size(Psi);
H2 = H * (H + 1) / 2;
V2 = V * (V - 1) / 2; % number of upper triangular elements in V x V matrix
[n, p] = size(x);
K = size(gamma1, 2); % rank K of tensor decomposition


%% saved quantities
beta_samples = zeros(L, H2, nsamples);
alpha_samples = zeros(H2, S, nsamples);
sigma2_tau_samples = zeros(nsamples, 1);
sigma2_alpha_samples = zeros(nsamples, 1);
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
mask_h_tri_V2 = repmat(triu(true(H)), 1, 1, V2); % (H, H, V2)
mask_h_tri_n = repmat(mask_h_tri, 1, 1, n); % (H, H, n), upper triangle element
mask_v_triu_n = repmat(triu(true(V, V), 1), 1, 1, n); % (V, V, n)
mask_v_triu_K = repmat(triu(true(V, V), 1), 1, 1, 1, K); % (V, V, 1, K)

% sigma2_e_shape = 1/2 + n * V * (V - 1) / 4;
sigma2_tau_shape = 1/2 + L * H * (H + 1) / 4;
sigma2_alpha_shape = 1/2 + S * H * (H + 1) / 4;

[~, clusters] = max(Z, [], 2);
H_old = H;
cluster_count = 0;


D = size(Lambda, 2); % number of factors in the factor analysis

%% iterations
tic;
for nn = 1:nsamples
    
    %% calculate \tilde{A}, \tilde{m}
    gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
    gamma2_x_reshape = reshape(x * gamma2, [1, 1, n, K]); % (1, 1, n, K)
    Gamma_x = sum(permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]) .* gamma2_x_reshape, 4); % (V, V, n)
    A_tilde = A - Gamma_x; % (V, V, n)
    A_tilde_triu = reshape(A_tilde(mask_v_triu_n), V2, n); % (V2, n)
    alpha_diag = permute(abs(alpha) > delta, [3, 2, 1]); % (1, S, H2)
    M_alpha = M .* alpha_diag; % (n, S, H2)
    Xi = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
    Xi_transpose_Xi = pagemtimes(Xi, 'transpose', Xi, 'none'); % (L, L, H2)
    Xi_beta = pagemtimes(Xi, reshape(beta, [L, 1, H2])); % (n, L, H2) x (L, 1, H2) = (n, 1, H2)
    Xi_beta_HtimesH = zeros(n, H, H);
    Xi_beta_HtimesH(:, mask_h_tri) = Xi_beta; % add upper tri & diag
    Xi_beta_HtimesH(:, mask_h_tril) = Xi_beta_HtimesH(:, mask_h_triu); % add lower tri
    m_tilde = pagemtimes(pagemtimes(Z, permute(Xi_beta_HtimesH, [2, 3, 1])), Z'); % (V, V, n)
    residual0 = A_tilde - m_tilde; % (V, V, n)
    residual0_triu = reshape(residual0(mask_v_triu_n), V2, n); % (V2, n)
    
    %% (1) beta_{h,h'}
    mask = triu(true(V), 1); % upper triangle mask
    zphiz = zeros(H, H);
    zphiA = zeros(H, H, n);
    z_check_ff = zeros(V2, H, H);
    z_check_phi = zeros(H, H, V2);
    for h2 = 1:H
        for h1 = 1:h2
            if h1 == h2
                z_check = (Z(:, h1) * Z(:, h2)');
            else
                z_check = (Z(:, h1) * Z(:, h2)' + Z(:, h2) * Z(:, h1)');
            end
            z_check = z_check(mask); % (V2, 1)
            z_check_ff(:, h1, h2) = z_check;
            z_check_phi_tmp = z_check' * Phi;
            z_check_phi(h1, h2, :) = z_check_phi_tmp;
            zphiz(h1, h2) = z_check_phi_tmp * z_check;
            zphiA(h1, h2, :) = z_check_phi_tmp * A_tilde_triu;
        end
    end
    zphiz_triu = reshape(zphiz(mask_h_tri), H2, 1); % (H2, 1)
    zphiA_triu = reshape(zphiA(mask_h_tri_n), H2, n); % (H2, n)
    z_check_ff_triu = reshape(z_check_ff(:, mask_h_tri), V2, H2); % (V2, H2)
    z_check_phi_triu = reshape(z_check_phi(mask_h_tri_V2), H2, V2); % (H2, V2)
    
    % mean and covariance for beta
    Sigma_beta_inv = reshape(zphiz_triu, 1, 1, H2) .* Xi_transpose_Xi ...
        + repmat(eye(L, L), 1, 1, H2) / sigma2_tau + ...
        repmat(Psi_lambda_square, 1, 1, H2) / sigma2_alpha; % (L, L, H2)
    Sigma_beta = pageinv(Sigma_beta_inv); % (L, L, H2)
    mu_beta2 = permute(alpha * Psi_lambda / sigma2_alpha, [2, 3, 1]); % (L, 1, H2)
    mu_beta2 = pagemtimes(Sigma_beta, mu_beta2); % (L, 1, H2)
    
    for h1_h2_idx = 1:H2
        if h1_h2_idx > 1
            tmp_old = tmp_curr;
            % residual0 = residual0 + tmp_old - tmp_new; % (V, V, n)
            residual0_triu = residual0_triu + tmp_old - tmp_new; % (V2, n)
        end
        Xi_beta_h1_h2 = Xi(:,:,h1_h2_idx) * beta(:, h1_h2_idx); % (n, L) x (L, 1) = (n, 1)
        tmp_curr = z_check_ff_triu(:, h1_h2_idx) .* Xi_beta_h1_h2'; % (V2, n)
        E_check_beta = residual0_triu + tmp_curr; % (V2, n)
        zphie_h1_h2 = z_check_phi_triu(h1_h2_idx, :) * E_check_beta; % (1, n)
        % calculate mu_beta
        mu_beta1 = Sigma_beta(:,:, h1_h2_idx) * (zphie_h1_h2 * Xi(:,:,h1_h2_idx))'; % (L, 1)
        mu_beta = mu_beta1 + mu_beta2(:, :, h1_h2_idx); % (L , 1)
        % sample from multivariate normal
        try
            beta(:, h1_h2_idx) = mvnrnd(mu_beta, Sigma_beta(:, :, h1_h2_idx)); % (L, 1)
        catch
            % fallback using regularization
            Sigma_beta_reg = nearestSPD(Sigma_beta(:, :, h1_h2_idx));
            beta(:, h1_h2_idx) = mvnrnd(mu_beta, Sigma_beta_reg);
        end
        
        Xi_beta_h1_h2 = Xi(:,:,h1_h2_idx) * beta(:, h1_h2_idx); % (n, L) x (L, 1) = (n, 1)
        tmp_new = z_check_ff_triu(:, h1_h2_idx) .* Xi_beta_h1_h2'; % (V2, n)
        % h1_h2_idx = h1_h2_idx + 1; % increment index
    end
    beta_samples(:, 1:H2, nn) = beta;
    
    
    %% (2) sample alpha(s_j), j = 1:S
    tau_tilde = reshape(Psi_lambda * beta, 1, S, H2); % (1, S, H2)
    tau_tilde_bound1 = (- delta - tau_tilde) / sqrt(sigma2_alpha); % (1, S, H2)
    tau_tilde_bound2 = (delta - tau_tilde) / sqrt(sigma2_alpha); % (1, S, H2)
    % sample from each truncated normal
    TN_neg = reshape(trandn(repelem(-Inf, S * H2), tau_tilde_bound1), [1, S, H2]); % (1, S, H2)
    TN_zero = reshape(trandn(tau_tilde_bound1, tau_tilde_bound2), [1, S, H2]); % (1, S, H2)
    TN_pos = reshape(trandn(tau_tilde_bound2, repelem(Inf, S * H2)), [1, S, H2]); % (1, S, H2)
    TN_all = cat(1, TN_neg, TN_zero, TN_pos) * sqrt(sigma2_alpha) + tau_tilde; % (3, S, H2)
    % calculate the weights
    pi_neg = normcdf(tau_tilde_bound1); % (1, S, H2)
    pi_zero = normcdf(tau_tilde_bound2) - normcdf(tau_tilde_bound1); % (1, S, H2)
    pi_pos = (1 - normcdf(tau_tilde_bound2)); % (1, S, H2)
    eps_prob = 1e-300;
    log_pi_neg = log(max(pi_neg, eps_prob));
    log_pi_zero = log(max(pi_zero, eps_prob));
    log_pi_pos = log(max(pi_pos, eps_prob));
    tau_tilde_M_sj = M .* tau_tilde; % (n, S, H2)
    c1 = sum(tau_tilde_M_sj .* permute(zphiA_triu, [2, 3, 1]), 1); % (n, S, H2) -> (1, S, H2)
    zphizff_triu2 = reshape(z_check_phi_triu * z_check_ff_triu, 1, H2, H2); % (1, H2, H2)
    g_rand = -log(-log(rand(3, S, H2)));
    % components_prob = rand(S, H2); % (S, H2)
    for j = 1:S
        c2 = sum(tau_tilde_M_sj(:, j, :) / 2 .* zphizff_triu2, 3); % (n, H2, H2) -> (n, H2), sum over f,f'
        for h_idx = 1:H2
            tau_M_sj = tau_tilde_M_sj .* alpha_diag; % (n, S, H2)
            tau_M_sum = sum(tau_M_sj, 2); % (n, 1, H2)
            tau_M_sum_non_sj = tau_M_sum - tau_M_sj(:, j, :); % (n, 1, H2)
            c3 = sum(tau_M_sum_non_sj .* zphizff_triu2(1, h_idx, :), 3); % (n, 1, H2) -> (n, 1), sum over f,f'
            c4 = tau_tilde_M_sj(:, j, h_idx)' * (c2(:, h_idx) + c3); % scalar
            log_c = c1(:, j, h_idx) - c4; % scalar
            
            log_prob = cat(1, log_pi_neg(1, j, h_idx) + log_c, log_pi_zero(1, j, h_idx), ...
                log_pi_pos(1, j, h_idx) + log_c); % (3, 1, 1)
            log_prob_max = max(log_prob, [], 1); % scalar
            log_prob_shift = log_prob - log_prob_max; % (3, 1)
            % decide which component to sample from
            [~, components_idx] = max(log_prob_shift + g_rand(:, j, h_idx), [], 1);
            alpha(h_idx, j) = TN_all(components_idx, j, h_idx); % scalar
            % update related quantities
            alpha_diag(1, j, h_idx) = abs(alpha(h_idx, j)) > delta; % (1, S, H2)
        end
    end
    alpha_samples(1:H2, :, nn) = alpha;
    
    M_alpha = M .* alpha_diag; % (n, S, H2)
    Xi = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
    Xi_beta = pagemtimes(Xi, reshape(beta, [L, 1, H2])); % (n, L, H2) x (L, 1, H2) = (n, 1, H2)
    Xi_beta_HtimesH = zeros(n, H, H);
    Xi_beta_HtimesH(:, mask_h_tri) = Xi_beta; % add upper tri & diag
    Xi_beta_HtimesH(:, mask_h_tril) = Xi_beta_HtimesH(:, mask_h_triu); % add lower tri
    
    
    % %% (3) update Phi_e
    m_tilde = pagemtimes(pagemtimes(Z, permute(Xi_beta_HtimesH, [2, 3, 1])), Z'); % (V, V, n)
    residual = A_tilde - m_tilde; % (V, V, n)
    residual_triu = reshape(residual(mask_v_triu_n), [], n)'; % (n, V2)
    
    % sample Lambda_d, d=1,...,D
    % Kappa: (n, D); Lambda: (V2, D);
    Kappa_square = Kappa' * Kappa; % (D, D)
    Kappa_e = Kappa' * residual_triu; % (D, V2)
    Sigma_Lambda = Kappa_square ./ reshape(sigma2_f_vec, 1, 1, V2) ...
        + eye(D) / sigma2_Lambda; % (D, D, V2)
    mu_Lambda = pagemldivide(Sigma_Lambda, reshape(Kappa_e ./ sigma2_f_vec, D, 1, V2)); % (D, 1, V2)
    Lambda = mvnrnd(permute(mu_Lambda, [3, 1, 2]), Sigma_Lambda); % (V2, D)
    
    % sample sigma2_f
    sigma2_f_shape = n / 2 + a_f;
    residual_triu_new = residual_triu -  Kappa * Lambda'; % (n, V2)
    sigma2_f_rate = sum(residual_triu_new.^2, 1) / 2 + 1 / b_f; % (1, V2)
    for j = 1:V2
        sigma2_f_vec(j) = 1 / gamrnd(sigma2_f_shape, 1 / sigma2_f_rate(j));
    end
    Sigma_f_inv = diag(1 ./ sigma2_f_vec); % (V2, V2)
    
    % sample Kappa
    Sigma_kappa_inv = Lambda' * Sigma_f_inv * Lambda + eye(D); % (D, D)
    Lambda_Sigma_f = Lambda' * Sigma_f_inv; % (D, V2)
    mu_Kappa_1 = Sigma_kappa_inv \ Lambda_Sigma_f; % (D, V2)
    mu_Kappa = mu_Kappa_1 * residual_triu'; % (D, n)
    Kappa = mvnrnd(mu_Kappa', Sigma_kappa_inv); % (n, D)
    
    % update Phi
    Phi = Sigma_f_inv - Lambda_Sigma_f' * mu_Kappa_1; % (V2, V2)
    
    
    %% sample sigma2_tau
    sigma2_tau_rate = sum(beta .* beta , [1, 2]) / 2 + 1 / a_tau;
    sigma2_tau = 1 / gamrnd(sigma2_tau_shape, 1 / sigma2_tau_rate); % scalar
    sigma2_tau_samples(nn) = sigma2_tau;
    
    %% sample sigma2_alpha
    sigma2_alpha_rate = sum((alpha' - reshape(tau_tilde, S, H2)).^2, "all") / 2 + 1 / a_alpha;
    sigma2_alpha = 1 / gamrnd(sigma2_alpha_shape, 1 / sigma2_alpha_rate); % scalar
    sigma2_alpha_samples(nn) = sigma2_alpha;
    
    % %% sample a_e, a_tau, a_alpha
    % % a_e = 1 / gamrnd(1, 1 / (sigma2_e^(-1) + b2_e^(-1)));
    % a_tau = 1 / gamrnd(1, 1 / (sigma2_tau^(-1) + b2_tau^(-1)));
    % a_alpha = 1 / gamrnd(1, 1 / (sigma2_alpha^(-1) + b2_alpha^(-1)));
    
    %% sample Zv, v = 1, ..., V
    % Get linear indices for upper-triangle (v1 < v2)
    [idx_row, idx_col] = find(triu(ones(V),1));
    M_tau_HtimesH = permute(Xi_beta_HtimesH, [2, 3, 1]); % (H, H, n)
    m_tilde_tt2 = pagemtimes(pagemtimes(Z, M_tau_HtimesH), Z'); % (V, V, n)
    m_tilde_vv_triu = reshape(m_tilde_tt2(mask_v_triu_n), V2, n); % (V2, n)
    
    tau_M_HtimesH_reshape = reshape(Xi_beta_HtimesH, [n, 1, H, H]); % (n, 1, H, H)
    tau_M_square = reshape(pagemtimes(tau_M_HtimesH_reshape, 'transpose',...
        tau_M_HtimesH_reshape, 'none'), H, H); % (1, 1, H, H) -> (H, H)
    
    for v = 1:V
        % Find edges involving & without node v (rows in Phi)
        mask_node_v = (idx_row == v) | (idx_col == v);
        edge_idx_tt2 = find(~mask_node_v); % indices of edges t, t', without node v, (V2 - (V-1), 1)
        edge_idx_vv2 = find(mask_node_v); % indices of edges involving node v', v' \neq v
        
        Phi_vv2_tt2 = Phi(edge_idx_vv2, edge_idx_tt2); % size: (V-1) x (V2 - (V-1)))
        Phi_vv2_vt2 = Phi(edge_idx_vv2, edge_idx_vv2); % size: (V-1) x (V-1)
        Phi_vv2 = Phi(edge_idx_vv2, :); % size: (V-1) x V2, all edges not involving v
        Z_minus_v = Z; Z_minus_v(v, :) = []; % (V-1, H), remove v-th node
        m_tilde_vv_triu_no_v = m_tilde_vv_triu(edge_idx_tt2, :); % (V2 - (V-1), n), remove v-th node
        s11 = Phi_vv2_tt2 * m_tilde_vv_triu_no_v; % (V-1, n)
        s12 = 2 * Phi_vv2 * A_tilde_triu; % (V-1, n)
        z_s1 = Z_minus_v' * (s11 - s12); % (H, n)
        z_s1 = sum(reshape(z_s1, [1, H, n]) .* M_tau_HtimesH, [2,3]); % (1, H, n) x (H, H, n) -> (H, H, n) ->  (H, 1)
        
        tau_M_square_Z = tau_M_square * Z_minus_v'; % (H, V-1)
        z_s21 = sum(tau_M_square_Z .* diag(Phi_vv2_vt2)', 2); % (H, V-1) -> (H, 1)
        z_s22_z = zeros(H, H);
        for v2 = 1:(V-1)
            for t2 = (v2+1):(V-1)
                z_s22_z = z_s22_z + Z_minus_v(v2,:)' * Phi_vv2_vt2(v2, t2) * Z_minus_v(t2, :);
            end
        end
        z_s22 = zeros(H, 1);
        for h = 1:H
            M_tau_h = reshape(M_tau_HtimesH(h, :, :), H, n); % (H, n)
            M_tau_h_h2f2 = M_tau_h * M_tau_h'; % (H, H), index for h2, f2
            % z_s22_h = 0;
            for h2 = 1:H
                for f2 = 1:H
                    z_s22(h) = z_s22(h) + z_s22_z(h2, f2) * M_tau_h_h2f2(h2, f2);
                end
            end
        end
        zvh_check = z_s1 + z_s21 + 2 * z_s22;
        
        log_pi_vh = log(w') - zvh_check / 2; % (H, 1)
        log_pi_vh_max = max(log_pi_vh); % scalar
        log_pi_vh_shift = log_pi_vh - log_pi_vh_max; % (H, 1)
        pi_z_shift = exp(log_pi_vh_shift); % (H, 1)
        pi_z_shift_norm = pi_z_shift / sum(pi_z_shift); % (H, 1)
        Z(v, :) = 0; % reset the v-th node
        Z(v, find(rand < cumsum(pi_z_shift_norm), 1)) = 1; % (H, 1)
    end
    Z_samples(:, 1:H, nn) = Z; % save samples
    
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
            
            beta = beta(:, H2_idx);
            alpha = alpha(H2_idx, :);
            Z = Z(:, ~zero_cluster); % (V, H)
            g = g(~zero_cluster); % (1, H)
            M_tau_HtimesH = M_tau_HtimesH(~zero_cluster, ~zero_cluster, :); % (H, H, n)
            % update the pre-calculated quantities
            mask_h_tri = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
            mask_h_tril = tril(true(H, H), -1);
            mask_h_triu = triu(true(H, H), 1);
            mask_h_tri_V2 = repmat(triu(true(H)), 1, 1, V2); % (H, H, V2)
            mask_h_tri_n = repmat(mask_h_tri, 1, 1, n); % (H, H, n), upper triangle element
        end
        
    end
    
    
    %% sample w
    w_concentration = sum(Z, 1) + g; % (1, H)
    w_gamma = gamrnd(w_concentration, 1); % (1, H)
    w = w_gamma / sum(w_gamma); % (1, H)
    w_samples(1:H, nn) = w;
    
    %% sample gamam_1vk, v=1,...,V, k=1,...,K
    gamma2_reshape = permute(gamma2, [3, 4, 1, 2]); % (1, 1, p, K)
    m_tilde = pagemtimes(pagemtimes(Z, M_tau_HtimesH), Z'); % (V, V, n)
    A_hat_gamma = A - m_tilde; % (V, V, n)
    x_gamma2 = x * gamma2; % (n, K)
    x_gamma2_square_sum = diag(x_gamma2' * x_gamma2); % (K, 1)
    for k = 1:K
        gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
        Gamma_K = permute(permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]) .* gamma2_reshape, [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
        Gamma_K(k, :, :, :) = 0; % remove the k-th component
        Gamma_x_non_k = permute(sum(pagemtimes(Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
        A_hat_gamma_k = A_hat_gamma - Gamma_x_non_k; % (V, V, n)
        A_hat_gamma_k_triu = reshape(A_hat_gamma_k(mask_v_triu_n), V2, n);
        Omega_vv2 = Phi * A_hat_gamma_k_triu; % (V2, n)
        for v = 1:V
            gamma1k_no_v = gamma1(:, k); gamma1k_no_v(v) = []; % (V-1, 1)
            mask_node_v = (idx_row == v) | (idx_col == v);
            edge_idx_vv2 = find(mask_node_v); % indices of edges involving node v', v' \neq v
            Phi_vv2_vt2 = Phi(edge_idx_vv2, edge_idx_vv2); % size: (V-1) x (V-1)
            gamma1k_no_v_square = gamma1k_no_v' * Phi_vv2_vt2 * gamma1k_no_v; % scalar
            sigma2_gamma1k_inv = x_gamma2_square_sum(k) * gamma1k_no_v_square + 1 / sigma2_gamma; % scalar
            sigma2_gamma1k = 1 / sigma2_gamma1k_inv; % scalar
            Omega_vv2_no_v = Omega_vv2(edge_idx_vv2, :); % (V-1, n)
            Omega_gamma1k_no_v = Omega_vv2_no_v' * gamma1k_no_v; % (n, 1)
            mu_gamma1k = x_gamma2(:, k).' * Omega_gamma1k_no_v / sigma2_gamma1k_inv; % scalar
            gamma1(v, k) = normrnd(mu_gamma1k, sqrt(sigma2_gamma1k));
        end
    end
    gamma1_samples(:, :, nn) = gamma1;
    
    %% sample gamma_2k, k=1,...,K
    gamma1_reshape = permute(gamma1, [1, 3, 2]); % (V, 1, K)
    gamma1_outer = permute(pagemtimes(gamma1_reshape, "none", gamma1_reshape, "transpose"), [1, 2, 4, 3]); % (V, V, 1, K)
    gamma1_check = reshape(gamma1_outer(mask_v_triu_K), [], K); % (V2, K), V2 = V*(V-1)/2
    gamma1_check_phi = gamma1_check' * Phi; % (K, V2)
    gamma1_check_square = diag(gamma1_check_phi * gamma1_check); % (K, K) -> (K, 1)
    x_square_sum = x' * x; % (p, p)
    for k = 1:K
        Sigma_gamma2k_inv = gamma1_check_square(k) * x_square_sum + eye(p) / sigma2_gamma; % (p, p)
        Sigma_gamma2k = inv(Sigma_gamma2k_inv); % (p, p)
        Gamma_K = permute(gamma1_outer .* permute(gamma2, [3, 4, 1, 2]), [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
        Gamma_K(k, :, :, :) = 0; % remove the k-th component
        Gamma_x_non_k = permute(sum(pagemtimes(Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
        A_hat_gamma_k = A_hat_gamma - Gamma_x_non_k; % (V, V, n)
        A_hat_gamma_k_check = reshape(A_hat_gamma_k(mask_v_triu_n), [], n); % (V2, n)
        mu_gamma2k = Sigma_gamma2k_inv \ (gamma1_check_phi(k, :) * A_hat_gamma_k_check * x)'; % (p, 1)
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
    sigma2_tau_samples = sigma2_tau_samples(eff_idx);
    sigma2_alpha_samples = sigma2_alpha_samples(eff_idx);
    Z_samples = Z_samples(:, :, eff_idx);
    w_samples = w_samples(:, eff_idx);
    gamma1_samples = gamma1_samples(:, :, eff_idx);
    gamma2_samples = gamma2_samples(:, :, eff_idx);
end

tau_tilde_samples = pagemtimes(Psi_lambda, beta_samples); % (S, H2, nsamples)
tau_tilde_samples = permute(tau_tilde_samples, [2, 1, 3]); % (S, H2, nsamples) -> (H2, S, nsamples)

end