%% data generation
function [A, M, x, Psi, Lambda_sqrt, B, BM, Gamma_x, ...
    tau, Z, w, gamma1, gamma2, sigma2_e, SNR_A, SNR_BM] ...
    = simu_data_gen_2d(n, V, H, p, s, sigma2_e, sigma2_gamma, delta, tau_pattern, ...
    seed, show_coeff, eigen_proportion, max_SNR, min_SNR)
% % Settings
% n = 100; % sample size
% V = 20; % number of nodes
% % S = 100; % number of vertices
% d = 2; % dimension of image predictor
% H = 3; % number of clusters
% H2 = H * (H + 1) / 2;
% % L = 20; % number of basis
% p = 2; % number of confounders
% sigma2_e = 0.1;
% sigma2_tau = 1;
% sigma2_alpha = 0.2;
% sigma2_gamma = 1.5;
% delta = 0.5;
if nargin < 12
    eigen_proportion = 0.95;
end

if isempty(seed) || ~exist("seed", "var")
    seed = 202412;
end
if isempty(show_coeff) || ~exist("show_coeff", "var")
    show_coeff = true;
end

H2 = H * (H + 1) / 2;

% image predictors
% s = 20;
sx = linspace(-1, 1, s);
sy = linspace(-1, 1, s);
[SX, SY] = meshgrid(sx, sy);
% figure; plot(SX(:), SY(:), '.');
Sv = [SX(:), SY(:)];
S = size(Sv, 1);
M0 = zeros(1, S);

% Compute the RBF kernel matrix
phi = 2; % Bandwidth parameter for RBF
nu = 2;
D = pdist2(Sv, Sv); % Pairwise squared distances
K = exp(- phi * D.^nu); % RBF kernel matrix
% Compute eigenvalues and eigenvectors
[eigvecs, eigvals_matrix] = eig(K); % Eigen decomposition
eigvals = diag(eigvals_matrix); % Extract eigenvalues
[~, idx] = sort(eigvals, 'descend'); % Sort eigenvalues and eigenvectors in descending order
eigvals = eigvals(idx);
eigvecs = eigvecs(:, idx);
eigvecs = eigvecs ./ vecnorm(eigvecs); % Normalize eigenvectors
[~ , L]  = max(cumsum(eigvals)/sum(eigvals) > eigen_proportion);
Psi = eigvecs(:, 1:L)'; % (L, S)
Lambda_sqrt = sqrt(eigvals(1:L)); % (L, 1)

% Coefficient
tau_tilde = zeros(S, H2);
for j = 1:H2
    if tau_pattern == 1
        tmp = sum((Sv - (j-H-1)/H).^2, 2)*j <= 0.3*j;
        tau_tilde(:,j) = -tmp .* Psi(1,:)' * 20;
    elseif tau_pattern == 2
        tmp = sum((Sv - (j-H-0.5)/H).^2, 2)*j <= 0.1*j;
        tau_tilde(:,j) = -tmp .* Psi(1,:)' * 20;
    elseif tau_pattern == 3
        % tmp = (s * j + 1 : s*(j+1));
        tmp = sum((Sv - (j-H-0.5)/H).^2, 2)*j <= 0.1*j;
        tau_tilde(:,j) = -tmp .* Psi(j,:)' * 80;
    elseif tau_pattern == 4
        tmp = sum((Sv - (j-0.8*H-1)*0.8/H).^2, 2)*j <= 0.3*j;
        tau_tilde(:,j) = -tmp .* Psi(1,:)' * 20;
    end
end
alpha = tau_tilde .* (abs(tau_tilde) > delta);
disp(mean(abs(alpha) > delta, 1))
tau = tau_tilde .* (abs(alpha) > delta);
if tau_pattern == 3
    tau = max(abs(tau), [], 1) .* sign(tau);
end
% form a H x H coefficient matrix T(s)
T = zeros(S, H, H);
mask_h = triu(true(H, H)); % mask for h,h', s.t., H >= h' >= h > 0
T(:, mask_h) = tau;
T = T + permute(T, [1, 3, 2]); % lower tri
T(:, 1:H+1:end) = T(:, 1:H+1:end) / 2; % adjust diagonal, double counting

if show_coeff
    figure('Name', 'True Coefficients', 'NumberTitle','off'); clf;
    sgtitle('\tau(s)')
    for j = 1:H2
        subplot(3,H2,j);
        imagesc([-1,1], [-1, 1], reshape(tau_tilde(:,j), [s, s])); colorbar; axis image;
        title(['$\tilde{\tau}_', num2str(j), '$'], 'Interpreter', 'latex');
        subplot(3,H2,j+H2);
        imagesc([-1, 1], [-1, 1], reshape(abs(alpha(:, j)), [s, s])); colorbar; axis image;
        title(['|\alpha_', num2str(j), '|']);
        subplot(3,H2,j+2*H2);
        imagesc([-1,1], [-1, 1], reshape(tau(:,j), [s, s])); colorbar; axis image;
        title(['\tau_', num2str(j)]);
    end
end


rng(seed);
M = mvnrnd(M0, K, n) / 10; % shape predictor, (n, S)

% other components
% membership
% pi_z = [0.2, 0.3, 0.5];
% w = rand(H, 1);
% w = w / sum(w);
w = [0.2; 0.3; 0.5]; % (H, 1)
Z = mnrnd(1, w, V);

% effect of predictors
B = pagemtimes(pagemtimes(Z, permute(T, [2, 3, 1])), Z'); % (V, V, S)
BM = permute(pagemtimes(M, permute(B, [3, 1, 2])), [2, 3, 1]); % (V, V, n)

% confounders
x = [ones(n,1), randn(n, p-1)]; % confounders, with intercept
% x = randn(n, p); % confounders, no intercept
gamma1 = randn(V, 1) * sqrt(sigma2_gamma);
gamma2 = randn(p, 1) * sqrt(sigma2_gamma);
Gamma_x = gamma1 .* gamma1' .* permute(x * gamma2, [3, 2, 1]); % (V, V, n)

% % connectivity matrix
% V2 = V * (V - 1) / 2;
mask_v = triu(true(V, V), 1); % (V, V)
mask_v_n = repmat(mask_v, 1, 1, n); % (V, V, n)
A_mean = Gamma_x + BM;
A_mean_triu = reshape(A_mean(mask_v_n), [], n); % (V2, n)
BM_triu = reshape(BM(mask_v_n), [], n); % (V2, n)


% random errors
% Errors = randn(V, V, n) * sigma2_e;
V2 = V * (V - 1) / 2;
errors = randn(V2, n) * sqrt(sigma2_e);
SNR_BM = abs(BM_triu ./ errors);
errors_idx = SNR_BM > max_SNR;
errors(errors_idx) = BM_triu(errors_idx) / max_SNR;
errors_idx = SNR_BM < min_SNR;
errors(errors_idx) = BM_triu(errors_idx) / min_SNR;

SNR_A = abs(A_mean_triu ./ errors);
SNR_BM = abs(BM_triu ./ errors);

% connectivity matrix
A_triu = A_mean_triu + errors;  % (V2, n)
A = zeros(V, V, n);
A(mask_v_n) = A_triu;
A = A + permute(A, [2, 1, 3]);

% mask_v = triu(true(V, V), 1); % (V, V)
% Errors = zeros(n, V, V);
% Errors(:, mask_v) = errors;
% Errors = permute(Errors + permute(Errors, [1, 3, 2]), [2, 3, 1]); % (V, V, n)

% % connectivity matrix
% A = Gamma_x + BM + Errors; % (V, V, n)


end
