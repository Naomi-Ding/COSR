function [fitted_errors] = simu_diagnosis_fitted(A, M, x, BM, B_hat, Gamma_hat)

% AUC_B_nodes: (1, V2)
% df_B_vtx_indicator: (1, 7) table
% df_B_indicator: (1, 7) table
% B_errors: (1, 6) table
% fitted_errors: (1, 6) table

if isempty(x) || sum(abs(x(:))) == 0
    confounder_vars = false;
else
    confounder_vars = true;
end

[V, ~, n] = size(A);
% [V, ~, S] = size(B);

%% R2 & MSE for response & BM
BM_hat = permute(pagemtimes(M, permute(B_hat, [3, 1, 2])), [2, 3, 1]); % (V, V, n)
if confounder_vars
    Gamma_x_hat = permute(pagemtimes(x, permute(Gamma_hat, [3, 1, 2])), [2, 3, 1]); % (V, V, n)
    A_hat = Gamma_x_hat + BM_hat;
else
    A_hat = BM_hat;
end
% n = size(A, 3);
mask_v = repmat(triu(true(V), 1), 1, 1, n);
A_hat_triu = reshape(A_hat(mask_v), [], n); % (V2, n)
A_triu = reshape(A(mask_v), [], n); % (V2, n)
if ~isempty(BM)
    BM_triu = reshape(BM(mask_v), [], n); % (V2, n)
    R2_hat_BM_all = zeros(1, n);
end
BM_hat_triu = reshape(BM_hat(mask_v), [], n); % (V2, n)
R2_hat_all = zeros(1, n);
for ii = 1:n
    R2_hat_all(ii) = corr(A_triu(:,ii), A_hat_triu(:,ii))^2;
    if ~isempty(BM)
        R2_hat_BM_all(ii) = corr(BM_triu(:,ii), BM_hat_triu(:,ii))^2;
    end
end
R2_hat = mean(R2_hat_all);
if ~isempty(BM)
    R2_hat_BM = mean(R2_hat_BM_all);
    BM_triu_diff_abs = abs(BM_triu - BM_hat_triu);
    MAE_BM_hat = mean(BM_triu_diff_abs, 'all');
    MSE_BM_hat = mean(BM_triu_diff_abs.^2, 'all');
else
    R2_hat_BM = NaN;
    MAE_BM_hat = NaN;
    MSE_BM_hat = NaN;
end
A_triu_diff_abs = abs(A_triu - A_hat_triu);
MAE_A_hat = mean(A_triu_diff_abs, 'all');
MSE_A_hat = mean(A_triu_diff_abs.^2, 'all');

fitted_errors = table(R2_hat, MAE_A_hat, MSE_A_hat, R2_hat_BM, MAE_BM_hat, MSE_BM_hat);


end