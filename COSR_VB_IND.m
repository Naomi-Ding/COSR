%% Variational Bayes update: interations of sample draw
function [beta_all, alpha_all, ...
    tau_all, sigma2_e_all, sigma2_tau_all, sigma2_alpha_all, ...
    Z_all, w_all, ELBO, gamma1_all, gamma2_all, Gamma_all, H, ...
    VBIC, convergence, E_I_abs_alpha_geq_delta] = ...
    COSR_VB_IND(A, M, x, Psi, Lambda_sqrt, ...
    mu_beta_ast, E_alpha, pi_alpha_neg, pi_alpha_zero, pi_alpha_pos, ...
    E_sigma2_e_inv, E_sigma2_tau_inv, E_sigma2_alpha_inv, E_Z, w_ast, ...
    mu_gamma1_ast, sigma2_gamma1_ast, mu_gamma2_ast, Sigma_gamma2_ast, ...
    E_a_e_inv, E_a_tau_inv, E_a_alpha_inv, delta, g, sigma2_gamma, niter, tol, ...
    fix_sigma2_alpha, correct_sigma2_e, check_point, shrink_cluster)
%%% Data (shapes):
%%%  A            (V, V, n)   - symmetric connectivity matrices (diagonal 0)
%%%  M            (n, S)      - shape measures evaluated on S spatial locations
%%%  x            (n, p)      - confounders / covariates
%%%  Psi          (L, S)      - basis/eigenfunctions (L basis x S locations)
%%%  Lambda_sqrt  (L, 1)      - sqrt eigenvalues (L x 1)

%%% Variational initials / parameters (shapes):
%%%  mu_beta_ast      (H2, L)  - mean of beta coefficients per unique (h,h') pair
%%%  E_alpha          (H2, S)  - E[alpha_{h,h'}(s)] for upper-triangular index pairs
%%%  pi_alpha_*       (H2, S)  - mixture probabilities (neg / zero / pos) per (h,h',s)
%%%  E_sigma2_*_inv   scalar   - expectations of inverse variances (e, tau, alpha)
%%%  E_Z              (V, H)   - variational cluster responsibilities (rows sum to 1)
%%%  w_ast            (1, H)   - Dirichlet pseudo-counts for cluster weights
%%%  mu_gamma1_ast    (V, K)   - variational means for node-specific gamma1
%%%  sigma2_gamma1_ast (V, K)  - variational variances for gamma1 (diagonal covariances)
%%%  mu_gamma2_ast    (p, K)   - variational means for gamma2
%%%  Sigma_gamma2_ast (p,p,K)  - variational covariances for gamma2
%%%  E_a_*_inv       scalar   - expectations of inverse variance hyper-parameters (e, tau, alpha)
%%%  E_b_*_inv       scalar   - expectations of inverse variance hyper-parameters (e, tau, alpha)

%%% Hyper-parameters and control:
%%%  delta            scalar  - threshold for alpha mixture
%%%  g                (1,H)   - Dirichlet concentration for cluster weights
%%%  sigma2_gamma     scalar  - prior variance for gamma components
%%%  niter, tol       - iterations and convergence tolerance

%%% Outputs (shapes):
%%%  beta_all         (L, H2) - posterior mean of beta coefficients per (h,h') pair
%%%  alpha_all        (H2, S) - posterior mean of alpha_{h,h'}(s) for upper-triangular index pairs
%%%  tau_all          (H2, S) - posterior mean of tau_{h,h'}(s) for upper-triangular index pairs
%%%  sigma2_*_all     scalar    - posterior means of variance parameters (e, tau, alpha)
%%%  Z_all            (V, H) - posterior mean of cluster responsibilities
%%%  w_all           (1, H) - posterior mean of cluster weights
%%%  gamma1_all      (V, K)  - posterior mean of node-specific gamma1
%%%  gamma2_all      (p, K)  - posterior mean of covariate gamma2
%%%  Gamma_all       (V, V, n) - posterior mean of regression term from confounders
%%%  H               scalar - final number of clusters


%%% Notes:
%%%  - H2 denotes the number of unique (h,h') upper-triangular pairs: H2 = H*(H+1)/2.
%%%  - This header documents shapes only; see function body for algorithm details.

if ~exist("niter", "var") || isempty(niter)
    niter = 1e3;
end
if ~exist("tol", "var") || isempty(tol)
    tol = 1e-3;
end

if ~exist("fix_sigma2_alpha", "var") || isempty(fix_sigma2_alpha)
    fix_sigma2_alpha = false;
end
if ~exist("correct_sigma2_e", "var") || isempty(correct_sigma2_e)
    correct_sigma2_e = true;
end

if ~exist("shrink_cluster", "var") || isempty(shrink_cluster)
    shrink_cluster = true;
end

n = size(A, 3);
[L, S] = size(Psi);
% [S, L] = size(Psi_lambda);
[V, H] = size(E_Z);
H2 = H * (H + 1) / 2;
H_old = H;

if isempty(x) || sum(abs(x(:))) == 0
    confounder_vars = false;
else
    confounder_vars = true;
    p = size(x, 2);
    K = size(mu_gamma1_ast, 2);
    mask_v_triu_K = repmat(triu(true(V, V), 1), 1, 1, 1, K); % (V, V, 1, K)
    mask_n_diag_K = repmat(diag(true(n, 1)), 1, 1, K); % (n, n, K)
    % mask_p_diag = reshape(repmat(diag(true(p, 1)), 1, 1, K), 1, p, p, K); % (1, p, p, K)
end


%% pre-calculate some quantities
Psi_lambda = Psi' .* Lambda_sqrt'; % (S, L)
Psi_lambda_square = Psi_lambda' * Psi_lambda; % (L, L)

E_I_abs_alpha_geq_delta = permute(pi_alpha_neg + pi_alpha_pos, [3, 2, 1]); % (1, S, H2)
M_alpha = M .* E_I_abs_alpha_geq_delta; % (n, S, H2)
Xi = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
Xi_square = pagemtimes(Xi, "transpose", Xi, "none"); % (L, L, H2)

b_e_ast = 1/2 + n * V * (V - 1) / 4; % unchanged
b_tau_ast = 1 / 2 + L * H * (H + 1) / 4; % unchanged
b_alpha_ast = 1 / 2 + S * H * (H + 1) / 4; % unchanged

mask_h_tri = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
mask_h_tril = tril(true(H, H), -1);
mask_h_triu = triu(true(H, H), 1);
mask_h_diag = repmat(diag(true(H, 1)), 1, 1, n); % (H, H, n), diagonal element
mask_n_diag_H2 = repmat(diag(true(n,1)), 1, 1, H2); % (n, n, H2)
mask_v_triu_n = repmat(triu(true(V, V), 1), 1, 1, n); % (V, V, n)
mask_v_triu_Hn = repmat(triu(true(V, V), 1), 1, 1, H, n); % (V, V, H, n), upper tri element
mask_s_diag = repmat(diag(true(S, 1)), 1, 1, H2); % (S, S, H2)

tau_tilde_ast = mu_beta_ast * Psi_lambda'; % (H2, S)
% tau_ast = tau_tilde_ast .* (E_I_abs_alpha_geq_delta > 0.5); % (H2, S)
tau_ast = tau_tilde_ast .* (abs(E_alpha) > delta); % (H2, S)
[~, clusters] = max(E_Z, [], 2);
binary_clusters = bsxfun(@eq, (1:H), clusters);
zero_cluster = sum(binary_clusters, 1) == 0; % (1, H)
H0 = sum(zero_cluster);
if H0 > 0
    tmp = mask_h_tri; % (H, H), upper tri
    tmp(zero_cluster, :) = 0;
    tmp(:, zero_cluster) = 0;
    H2_idx = tmp(mask_h_tri);
    tau_ast = tau_ast(H2_idx, :); % (H2, S)
end

ELBO = -Inf;
VBIC = Inf;
cluster_count = 0;

convergence = false;

tic;
for nn = 1:niter
    
    if confounder_vars
        %% (1) update mu_gamma1kv & sigma2_gamma1kv for v=1,...,V, k=1,...,K
        mu_gamma2_reshape = permute(mu_gamma2_ast, [3, 4, 1, 2]); % (p, K) -> (1, 1, p, K)
        E_gamma2_x_square_1 = pagemtimes(pagemtimes(x, Sigma_gamma2_ast), x'); % (n, n, K)
        E_gamma2_x_square_1_sum = sum(reshape(E_gamma2_x_square_1(mask_n_diag_K), n, K), 1); % (1, K)
        mu_gamma2_x = x * mu_gamma2_ast; % (n, K)
        E_gamma2_x_square_2_sum = diag(mu_gamma2_x' * mu_gamma2_x); % (K, 1)
        E_gamma2_x_square_sum = E_gamma2_x_square_1_sum' + E_gamma2_x_square_2_sum; % (K, 1)
        mu_beta_ast_reshape = permute(mu_beta_ast, [2, 3, 1]); %  (L, 1, H2)
        Xi_mu_beta = pagemtimes(Xi, mu_beta_ast_reshape); % (n, 1, H2)
        Xi_mu_beta_reshape = zeros(n, H, H);
        Xi_mu_beta_reshape(:, mask_h_tri) = Xi_mu_beta; % add upper tri
        Xi_mu_beta_reshape(:, mask_h_tril) = Xi_mu_beta_reshape(:, mask_h_triu);  % add lower tri
        Z_Xi_mu_beta_Z = pagemtimes(pagemtimes(E_Z, permute(Xi_mu_beta_reshape, ...
            [2, 3, 1])), E_Z'); % (V, V, n), m_tilde
        A_hat_gamma = A - Z_Xi_mu_beta_Z; % (V, V, n)
        for k = 1:K
            mu_gamma1_reshape = permute(mu_gamma1_ast, [1, 3, 2]); % (V, 1, K)
            E_Gamma_K = permute(permute(pagemtimes(mu_gamma1_reshape, "none", ...
                mu_gamma1_reshape, "transpose"), [1, 2, 4, 3]) .* mu_gamma2_reshape, ...
                [4, 3, 1, 2]); % (K, p, V, V)
            E_Gamma_K(k, :, :, :) = 0; % remove the k-th component
            E_Gamma_x_non_k = permute(sum(pagemtimes(E_Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
            E_A_hat_gamma_k = A_hat_gamma - E_Gamma_x_non_k; % (V, V, n)
            for v = 1:V
                mu_gamma1_ast(v, k) = 0; % (V, K), remove the v-th node
                sigma2_gamma1_ast(v, k) = 0; % (V, K), remove the v-th node
                E_gamma1k_no_v_square = sum(sigma2_gamma1_ast(:, k)) ...
                    + mu_gamma1_ast(:, k)' * mu_gamma1_ast(:, k); % scalar
                sigma2_gamma1k_inv = E_gamma2_x_square_sum(k) * E_gamma1k_no_v_square ...
                    * E_sigma2_e_inv + 1 / sigma2_gamma; % scalar
                sigma2_gamma1_ast(v, k) = 1 / sigma2_gamma1k_inv;
                E_A_hat_gamma_k_no_v = reshape(E_A_hat_gamma_k(v, :, :), V, n); % (V, n)
                E_A_hat_gamma_k_no_v(v, :) = 0; % remove the v-th node
                mu_gamma1_ast(v, k) = mu_gamma2_x(:, k)' * E_A_hat_gamma_k_no_v' ...
                    * mu_gamma1_ast(:, k) * E_sigma2_e_inv * sigma2_gamma1_ast(v, k);
            end
        end
        
        
        %% (2) update mu_gamma2k & Sigma_gamma2k for k=1,...,K
        mu_gamma1_reshape = permute(mu_gamma1_ast, [1, 3, 2]); % (V, 1, K)
        mu_gamma1_outer = permute(pagemtimes(mu_gamma1_reshape, "none", ...
            mu_gamma1_reshape, "transpose"), [1, 2, 4, 3]); % (V, V, 1, K)
        E_gamma1_check = reshape(mu_gamma1_outer(mask_v_triu_K), [], K); % (V2, K), V2=V*(V-1)/2
        E_gamma1_check_square_1 = permute(sigma2_gamma1_ast + mu_gamma1_ast.^2, [1, 3, 2]); % (V, 1, K)
        E_gamma1_check_square_1_outer = permute(pagemtimes(E_gamma1_check_square_1, ...
            "none", E_gamma1_check_square_1, "transpose"), [1, 2, 4, 3]); % (V, V, 1, K)
        E_gamma1_check_square = sum(reshape(E_gamma1_check_square_1_outer(mask_v_triu_K), ...
            [], K), 1); % (1, K)
        x_square_sum_sigma2e_inv = x' * x * E_sigma2_e_inv; % (p, p)
        for k = 1:K
            Sigma_gamma2k_inv = E_gamma1_check_square(k) * x_square_sum_sigma2e_inv ...
                + eye(p) / sigma2_gamma; % (p, p)
            Sigma_gamma2_ast(:, :, k) = inv(Sigma_gamma2k_inv); % (p, p)
            % Sigma_gamma2_ast_logdet(k) = logdet(Sigma_gamma2_ast(:,:,k), "chol"); % scalar
            E_Gamma_K = permute(mu_gamma1_outer .* permute(mu_gamma2_ast, [3, 4, 1, 2]), ...
                [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
            E_Gamma_K(k, :, :, :) = 0; % remove the k-th component
            E_Gamma_x_non_k = permute(sum(pagemtimes(E_Gamma_K, x'), 1), [3, 4, 2, 1]); % (V, V, n, 1)
            E_A_hat_gamma_k = A_hat_gamma - E_Gamma_x_non_k; % (V, V, n)
            E_A_hat_gamma_k_check = reshape(E_A_hat_gamma_k(mask_v_triu_n), [], n); % (V2, n)
            mu_gamma2_ast(:, k) = Sigma_gamma2k_inv \ (E_gamma1_check(:, k)' ...
                * E_A_hat_gamma_k_check * x)' * E_sigma2_e_inv;
        end
        E_Gamma = permute(mu_gamma1_outer .* permute(mu_gamma2_ast, [3, 4, 1, 2]), ...
            [4, 3, 1, 2]); % (V, V, p, K) -> (K, p, V, V)
        E_Gamma_x = permute(sum(pagemtimes(E_Gamma, x'), 1), [3, 4, 2, 1]); % (V, V, n)
        
    else
        E_Gamma_x = 0;
    end
    
    %% (3) calculate \sum_{v<v'}ZvZv' & \sum_{v<v'}ZvZv'Avv'
    A_tilde = A - E_Gamma_x; % (V, V, n)
    % sum_{V >= v' > v > 0}(E[Z_{vh}]E[Z_{v'h'}])
    Zh = sum(E_Z, 1); % (1, H)
    Zh2 = Zh' * Zh; % (H, H)
    Zh_diag = Zh .* (Zh - 1) / 2; % (1, H)
    ZvZv_sum = diag(Zh_diag) + triu(Zh2, 1); % (H, H)
    ZvZv_sum_triu = reshape(ZvZv_sum(mask_h_tri), 1, 1, H2); % (1, 1, H2)
    
    % sum_{V >= v' > v > 0}(E[Z_{vh}]E[Z_{v'h'}] A_tilde)
    ZAZ = pagemtimes(pagemtimes(E_Z', A_tilde), E_Z); % (H, H, n)
    ZAZ_diag = permute(E_Z, [1, 3, 2]) .* permute(E_Z, [3, 1, 2])...
        .* permute(A_tilde, [1, 2, 4, 3]); % (V, V, H, n)
    ZAZ(mask_h_diag) = sum(reshape(ZAZ_diag(mask_v_triu_Hn), [], H, n), 1);
    ZAZ_triu = permute(reshape(ZAZ(repmat(mask_h_tri, 1, 1, n)), H2, n), [3, 2, 1]); % (H2, n) -> (1, n, H2)
    
    
    %% (5) update mu*_beta & Sigma*_beta
    % mu_beta_ast_old = mu_beta_ast; % (H2, L)
    Sigma_beta_inv = ZvZv_sum_triu .* Xi_square * E_sigma2_e_inv + ...
        repmat(eye(L, L), 1, 1, H2) * E_sigma2_tau_inv + ...
        repmat(Psi_lambda_square, 1, 1, H2) * E_sigma2_alpha_inv; % (L, L, H2)
    Sigma_beta_ast = pageinv(Sigma_beta_inv); % (L, L, H2)
    mu_beta1 =  permute(pagemtimes(ZAZ_triu, Xi), [2, 1, 3]) * E_sigma2_e_inv ...
        + permute(E_alpha * Psi_lambda, [2, 3, 1]) * E_sigma2_alpha_inv; % (L, 1, H2)
    mu_beta_ast_reshape = pagemtimes(Sigma_beta_ast, mu_beta1); % (L, 1, H2)
    mu_beta_ast = permute(mu_beta_ast_reshape, [3, 1, 2]); % (H2, L, 1)
    
    
    %% (4) update mu_alpha_ast & sigma2_alpha_ast & pi_alpha
    mu_alpha_ast = mu_beta_ast * Psi_lambda'; % (H2, S)
    sigma2_alpha_ast = 1 / E_sigma2_alpha_inv; % scalar
    sigma_alpha = sqrt(sigma2_alpha_ast); % scalar
    
    % E_tau_tilde = mu_alpha_ast; % (H2, S)
    mu_beta_square = pagemtimes(mu_beta_ast_reshape, "none", mu_beta_ast_reshape, "transpose"); % (L, L, H2)
    E_tau_tilde_square = pagemtimes(pagemtimes(Psi_lambda, Sigma_beta_ast ...
        + mu_beta_square), Psi_lambda'); % (S, S, H2)
    tau_tilde_M_sj = M .* permute(mu_alpha_ast, [3, 2, 1]); % (n, S, H2)
    c1 = reshape(pagemtimes(ZAZ_triu, tau_tilde_M_sj), S, H2) * E_sigma2_e_inv; % (S, H2)
    E_tau_tilde_square_diag = reshape(E_tau_tilde_square(mask_s_diag), [S, H2]); % (S, H2)
    tau_tilde_square_M_sj = diag(M' * M) .* E_tau_tilde_square_diag / 2; % (S, H2)
    c2 = reshape(ZvZv_sum_triu, 1, []) .* tau_tilde_square_M_sj * E_sigma2_e_inv; % (S, H2)
    c3a = tau_tilde_M_sj .* ZvZv_sum_triu * E_sigma2_e_inv; % (n, S, H2)
    
    % pi_alpha
    delta_neg = (- delta - mu_alpha_ast) / sigma_alpha; % delta^(-1), (H2, S)
    delta_pos = (delta - mu_alpha_ast) / sigma_alpha; % delta^(1), (H2, S)
    Phi_delta_neg = normcdf(delta_neg); % (H2, S)
    Phi_delta_neg(Phi_delta_neg == 0) = 1e-32;
    Phi_delta_pos = normcdf(delta_pos); % (H2, S)
    Phi_delta_pos_comp = max(1 - Phi_delta_pos, normcdf(-delta_pos)); % (H2, S)
    Phi_delta_pos_comp(Phi_delta_pos_comp == 0) = 1e-32;
    Phi_delta_mid = Phi_delta_pos - Phi_delta_neg; % (H2, S)
    Phi_delta_mid(Phi_delta_mid == 0) = 1e-32;
    for j = 1:S
        % update c = c1 + c2 + c3
        tau_M_sj = tau_tilde_M_sj .* E_I_abs_alpha_geq_delta; % (n, S, H2)
        tau_M_sj(:, j, :) = 0;
        tau_M_sum_non_sj = sum(tau_M_sj, 2); % (n, 1, H2)
        c3 = reshape(sum(c3a(:, j, :) .* tau_M_sum_non_sj, 1), 1, H2); % (1, H2)
        log_c = c1(j, :) - c2(j, :) - c3; % (1, H2)
        
        c = exp(log_c);
        % update pi_alpha_neg, pi_alpha_zero, pi_alpha_pos at s_j
        pi_neg_sj = Phi_delta_neg(:, j); % (H2, 1)
        pi_pos_sj = Phi_delta_pos_comp(:, j); % (H2, 1)
        pi_zero_sj = Phi_delta_mid(:, j) ./ c'; % (H2, 1)
        pi_sum_sj = pi_neg_sj + pi_pos_sj + pi_zero_sj; % (H2, 1)
        pi_alpha_neg(:, j) = pi_neg_sj ./ pi_sum_sj;
        pi_alpha_zero(:,j) = pi_zero_sj ./ pi_sum_sj;
        pi_alpha_pos(:, j) = pi_pos_sj ./ pi_sum_sj;
        if any(isinf(pi_zero_sj)) % inf, c~0
            pi_alpha_zero(isinf(pi_zero_sj), j) = 1;
        end
        % update E[I(|alpha_{h,h',j}| > delta)]
        E_I_abs_alpha_geq_delta(1, j, :) = pi_alpha_neg(:, j) + pi_alpha_pos(:,j);
    end
    
    % E[alpha_{h,h'}(s_j)] & E[(alpha_{h,h'}(s_j))^2] & E[I(|alpha_{h,h',j}| > delta)]
    phi_delta_neg = normpdf(delta_neg); % (H2, S)
    phi_delta_pos = normpdf(delta_pos); % (H2, S)
    phi_vs_Phi_delta_neg = phi_delta_neg ./ Phi_delta_neg; % (H2, S)
    phi_vs_Phi_delta_mid = (phi_delta_pos - phi_delta_neg) ./ Phi_delta_mid; % (H2, S)
    phi_vs_Phi_delta_pos = phi_delta_pos ./ Phi_delta_pos_comp; % (H2, S)
    E_alpha_neg = mu_alpha_ast - sigma_alpha * phi_vs_Phi_delta_neg; % E[alpha_{h,h',j} | alpha_{h,h',j} < -delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
    E_alpha_mid = mu_alpha_ast - sigma_alpha * phi_vs_Phi_delta_mid; % E[alpha_{h,h',j} | |alpha_{h,h',j}| < delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
    E_alpha_pos = mu_alpha_ast + sigma_alpha * phi_vs_Phi_delta_pos; % E[alpha_{h,h',j} | alpha_{h,h',j} > delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
    E_alpha = pi_alpha_neg .* E_alpha_neg + pi_alpha_zero .* E_alpha_mid + ...
        pi_alpha_pos .* E_alpha_pos; % E[alpha_{h,h',j}] for  1<=h<=h'<=H, j=1,...,S, (H2, S)
    
    % update Xi & Xi^\top X_i
    M_alpha = M .* E_I_abs_alpha_geq_delta; % (n, S, H2)
    Xi = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
    Xi_square = pagemtimes(Xi, "transpose", Xi, "none"); % (L, L, H2)
    
    
    %% (6) update b_e_ast, d_e_ast
    if confounder_vars
        d_e_ast_1 = sum(A.^2 + (E_Gamma_x - 2 * A) .* E_Gamma_x, 3); % (V, V)
        E_gamma1k_square = permute(sigma2_gamma1_ast + mu_gamma1_ast.^2, [1, 3, 2]); % (V, 1, K)
        E_gamma1k_square_v_vprime = permute(pagemtimes(E_gamma1k_square, 'none', ...
            E_gamma1k_square, 'transpose'), [1, 2, 4, 3]); % (V, V, 1, K)
        mu_gamma2_ast_reshape = permute(mu_gamma2_ast, [1, 3, 2]); % (p, K) -> (p, 1, K)
        mu_gamma2_ast_square = pagemtimes(mu_gamma2_ast_reshape, ...
            'none', mu_gamma2_ast_reshape, 'transpose'); % (p, p, K)
        E_gamma2k_square = permute(Sigma_gamma2_ast + mu_gamma2_ast_square, [4, 1, 2, 3]); % (1, p, p, K)
        square_E_gamma1k = permute(mu_gamma1_ast.^2, [1, 3, 2]); % (V, 1, K)
        square_E_gamma1k_v_vprime = permute(pagemtimes(square_E_gamma1k, 'none',...
            square_E_gamma1k, 'transpose'), [1, 2, 4, 3]); % (V, V, 1, K)
        Var_Gamma = sum(reshape(E_gamma1k_square_v_vprime(mask_v_triu_K), [], 1,...
            1, K) .* E_gamma2k_square - reshape(square_E_gamma1k_v_vprime(...
            mask_v_triu_K), [], 1, 1, K) .* permute(mu_gamma2_ast_square, ...
            [4, 1, 2, 3]), [1, 4]); % (V2, p, p, K) -> (1, p, p)
        X_Var_Gamma_X = sum(diag(x * reshape(Var_Gamma, p, p) * x')); % (n, n) -> scalar
    else
        d_e_ast_1 = sum(A.^2, 3); % (V, V)
        X_Var_Gamma_X = 0;
    end
    
    Xi_mu_beta = pagemtimes(Xi, mu_beta_ast_reshape); % (n, 1, H2)
    Xi_mu_beta_square = reshape(pagemtimes(Xi_mu_beta, "transpose", Xi_mu_beta, "none"), H2, 1); % (H2, 1)
    Xi_mu_beta_square_reshape = zeros(H, H);
    Xi_mu_beta_square_reshape(mask_h_tri) = Xi_mu_beta_square; % add upper tri
    Xi_mu_beta_square_reshape(mask_h_tril) = Xi_mu_beta_square_reshape(mask_h_triu); % add lower tri
    d_e_ast_2 = E_Z * Xi_mu_beta_square_reshape * E_Z'; % (V, V)
    Xi_mu_beta_reshape = zeros(n, H, H);
    Xi_mu_beta_reshape(:, mask_h_tri) = Xi_mu_beta; % add upper tri
    Xi_mu_beta_reshape(:, mask_h_tril) = Xi_mu_beta_reshape(:, mask_h_triu);  % add lower tri
    Z_Xi_mu_beta_Z = pagemtimes(pagemtimes(E_Z, permute(Xi_mu_beta_reshape, ...
        [2, 3, 1])), E_Z'); % (V, V, n)
    
    if ~correct_sigma2_e
        d_e_ast_3 = 2 * sum(A_tilde .* Z_Xi_mu_beta_Z, 3); % (V, V)
    else
        d_e_ast_3 = - 2 * sum(A_tilde .* Z_Xi_mu_beta_Z, 3); % (V, V)
    end
    
    d_e_ast = E_a_e_inv + sum(triu(d_e_ast_1 + d_e_ast_2 + d_e_ast_3, 1), "all") / 2 ...
        + X_Var_Gamma_X / 2;
    
    % update E_sigma2_e_inv
    E_sigma2_e_inv = b_e_ast / d_e_ast;
    
    
    %% (7) update b_tau_ast, d_tau_ast
    mask_l_diag = repmat(diag(true(L, 1)), 1, 1, H2); % (L, L, H2)
    d_tau_ast = E_a_tau_inv + sum(reshape(Sigma_beta_ast(mask_l_diag), ...
        [L, H2]), "all") / 2 + sum(diag(mu_beta_ast * mu_beta_ast')) / 2;
    
    % update E_sigma2_tau_inv
    E_sigma2_tau_inv = b_tau_ast / d_tau_ast;
    
    
    %% (8) update b_alpha_ast, d_alpha_ast
    if ~fix_sigma2_alpha
        Var_alpha_neg = sigma2_alpha_ast * ( 1 - delta_neg .* phi_vs_Phi_delta_neg ...
            - phi_vs_Phi_delta_neg.^2 ); % Var[alpha_{h,h',j} | alpha_{h,h',j} < -delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
        Var_alpha_mid = sigma2_alpha_ast * (1 - (delta_pos .* phi_delta_pos - ...
            delta_neg .* phi_delta_neg) ./ Phi_delta_mid - phi_vs_Phi_delta_mid.^2 ); % Var[alpha_{h,h',j} | |alpha_{h,h',j}| < delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
        Var_alpha_pos = sigma2_alpha_ast * (1 + delta_pos .* phi_vs_Phi_delta_pos ...
            - phi_vs_Phi_delta_pos.^2 ); % Var[alpha_{h,h',j} | alpha_{h,h',j} > delta] for 1<=h<=h'<=H, j=1,...,S, (H2, S)
        E_alpha_square = pi_alpha_neg .* (Var_alpha_neg + E_alpha_neg.^2) ...
            + pi_alpha_zero .* (Var_alpha_mid + E_alpha_mid.^2) ...
            + pi_alpha_pos .* (Var_alpha_pos + E_alpha_pos.^2); % E[(alpha_{h,h',j})^2] for  1<=h<=h'<=H, j=1,...,S, (H2, S)
        d_alpha_ast = E_a_alpha_inv + sum(E_alpha_square - 2 * E_alpha .* mu_alpha_ast ...
            + E_tau_tilde_square_diag', "all") / 2; % (H2, S) -> scalar
        
        % update E_sigma2_alpha_inv
        E_sigma2_alpha_inv = b_alpha_ast / d_alpha_ast;
    end
    
    %% (9) update E_Zv, v=1,...,V
    E_Z_old = E_Z;
    E_w_log = psi(w_ast) - psi(sum(w_ast)); % (1, H)
    pi_z_1 = pagemtimes(permute(A_tilde, [2, 3, 4, 1]), Xi_mu_beta_reshape); % (V, H, H, V)
    Xi_Sigma_beta_Xi = pagemtimes(pagemtimes(Xi, Sigma_beta_ast), ...
        "none", Xi, "transpose"); % (n, n, H2)
    Xi_Sigma_beta_Xi_diag = reshape(Xi_Sigma_beta_Xi(mask_n_diag_H2), [n, H2]); % (n, H2)
    Xi_Sigma_beta_Xi_sum = sum(Xi_Sigma_beta_Xi_diag, 1); % (1, H2)
    pi_z_2 = Xi_Sigma_beta_Xi_sum + Xi_mu_beta_square'; % (1, H2)
    pi_z_2_reshape = zeros(H, H);
    pi_z_2_reshape(mask_h_tri) = pi_z_2; % upper tri
    pi_z_2_reshape(mask_h_tril) = pi_z_2_reshape(mask_h_triu); % lower tri
    for v = 1:V
        E_Z(v, :) = 0; % (V, H)
        pi_z_1_sum = reshape(sum(pi_z_1(:, :, :, v) .* E_Z, [1, 2]), H, 1); % (H, 1), sum over v' \neq v & h', for v'=1-V, h'=1-H
        pi_z_2_sum = sum(E_Z * pi_z_2_reshape, 1) / 2; % (1, H), sum over v' \neq v & h', for v'=1-V, h'=1-H
        log_pi_z = (pi_z_1_sum' - pi_z_2_sum) * E_sigma2_e_inv + E_w_log; % (1, H)
        log_pi_z_shift = log_pi_z - max(log_pi_z); % (1, H)
        pi_z_shift = exp(log_pi_z_shift); % (1, H)
        pi_z_shift_normalized = pi_z_shift / sum(pi_z_shift); % (1, H)
        E_Z(v, :) = pi_z_shift_normalized;
    end
    
    %% check whether to shrink the clusters
    if shrink_cluster
        clusters_old = clusters; % (V, 1)
        [~, clusters] = max(E_Z, [], 2);
        clusters_diff = sum(abs(clusters - clusters_old));
        if clusters_diff == 0 % all the node memberships are the same with the previous iteration
            cluster_count = cluster_count + 1;
        else
            cluster_count = 0; % restart the counting if the node memberships are different
        end
        if ((max(abs(E_Z - E_Z_old), [], 'all') < tol) || (cluster_count >= 3))
            % [~, clusters] = max(E_Z, [], 2);
            binary_clusters = bsxfun(@eq, (1:H), clusters); % (V, H)
            cluster_prop = mean(binary_clusters, 1); % (1, H)
            zero_cluster = sum(binary_clusters, 1) == 0; % (1, H)
            % zero_cluster = zero_cluster | (cluster_prop <= 0.05); % (1, H), set clusters with proportion <= 0.05 to empty
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
                pi_alpha_neg = pi_alpha_neg(H2_idx, :); % (H2, S)
                pi_alpha_zero = pi_alpha_zero(H2_idx, :);
                pi_alpha_pos = pi_alpha_pos(H2_idx, :); % (H2, S)
                E_I_abs_alpha_geq_delta = permute(pi_alpha_neg + pi_alpha_pos, [3, 2, 1]); % (1, S, H2)
                M_alpha = M .* E_I_abs_alpha_geq_delta; % (n, S, H2)
                Xi = pagemtimes(M_alpha, Psi_lambda); % (n, L, H2)
                Xi_square = pagemtimes(Xi, "transpose", Xi, "none"); % (L, L, H2)
                mu_beta_ast = mu_beta_ast(H2_idx, :); % (H2, L)
                E_alpha = E_alpha(H2_idx, :); % (H2, S)
                tau_ast = tau_ast(H2_idx, :); % (H2, S)
                % Sigma_beta_ast = Sigma_beta_ast(:,:, H2_idx); % (L, L, H2)
                % Phi_delta_neg = Phi_delta_neg(H2_idx, :);
                % Phi_delta_mid = Phi_delta_mid(H2_idx, :);
                % Phi_delta_pos_comp = Phi_delta_pos_comp(H2_idx, :);
                
                % update the pre-calculated quantities
                mask_h_tri = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
                mask_h_tril = tril(true(H, H), -1);
                mask_h_triu = triu(true(H, H), 1);
                mask_h_diag = repmat(diag(true(H, 1)), 1, 1, n); % (H, H, n), diagonal element
                mask_n_diag_H2 = repmat(diag(true(n,1)), 1, 1, H2); % (n, n, H2)
                mask_v_triu_Hn = repmat(triu(true(V, V), 1), 1, 1, H, n); % (V, V, H, n), upper tri element
                mask_s_diag = repmat(diag(true(S, 1)), 1, 1, H2); % (S, S, H2)
                b_tau_ast = 1 / 2 + L * H * (H + 1) / 4; % unchanged
                b_alpha_ast = 1 / 2 + S * H * (H + 1) / 4; % unchanged
                g = g(~zero_cluster); % (1, H)
                
                E_Z = E_Z(:, ~zero_cluster); % (V, Hnew)
                E_Z(sum(E_Z, 2) == 0, :) = 1 / H;
                E_Z = E_Z ./ sum(E_Z, 2);
                E_Z_old = E_Z_old(:, ~zero_cluster); % (V, Hnew)
                E_Z_old(sum(E_Z_old, 2) == 0, :) = 1 / H;
                E_Z_old = E_Z_old ./ sum(E_Z_old, 2);
            end
            
        end
    end
    
    %% tau hat
    tau_ast_old = tau_ast;
    tau_tilde_ast = mu_beta_ast * Psi_lambda'; % (H2, S)
    tau_ast = tau_tilde_ast .* (abs(E_alpha) > delta); % (H2, S)
    
    
    %% (10) update w_ast
    w_ast = sum(E_Z, 1) + g; % (1, H)
    
    %% save the last one
    
    if confounder_vars
        gamma1_all = mu_gamma1_ast;
        gamma2_all = mu_gamma2_ast;
        Gamma_all = permute(sum(E_Gamma, 1), [3, 4, 2, 1]);
    end
    beta_all = mu_beta_ast;
    alpha_all = E_alpha;
    sigma2_e_all = d_e_ast / (b_e_ast - 1);
    sigma2_tau_all = d_tau_ast / (b_tau_ast - 1);
    tau_all = tau_ast;
    
    if ~fix_sigma2_alpha
        sigma2_alpha_all = d_alpha_ast / (b_alpha_ast - 1);
    end
    Z_all = E_Z;
    
    
    %% check the stopping criterion
    if mod(nn, 100) == 0
        disp(nn);
        toc;
    end
    
    if exist("check_point", "var") && (~isempty(check_point)) && mod(nn, 10) == 0 % save current estimates
        w_0 = w_ast;
        save(check_point, "w_0", "E_Z", "mu_beta_ast", "E_alpha", "pi_alpha_zero",...
            "pi_alpha_neg", "pi_alpha_pos", "E_sigma2_e_inv", "E_sigma2_tau_inv", ...
            "E_sigma2_alpha_inv");
        if confounder_vars
            save(check_point, "mu_gamma1_ast", "mu_gamma2_ast", ...
                "sigma2_gamma1_ast", "Sigma_gamma2_ast", "-append");
        end
    end
    
    % if abs(ELBO - ELBO_old) < tol
    if (max(abs(E_Z - E_Z_old), [], 'all') < tol) && ...
            (max(abs(tau_ast - tau_ast_old), [], 'all') < tol)
        disp(['Converged after ', num2str(nn), ' iterations']);
        convergence = true;
        toc;
        
        break;
    end
    
end


end