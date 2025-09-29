function [AUC_B_nodes, df_B_vtx_indicator, df_B_indicator, ...
    B_errors] = simu_diagnosis_estimation_B(B, B_hat)

% AUC_B_nodes: (1, V2)
% df_B_vtx_indicator: (1, 7) table
% df_B_indicator: (1, 7) table
% B_errors: (1, 6) table
% fitted_errors: (1, 6) table

% if isempty(x) || sum(abs(x(:))) == 0
%     confounder_vars = false;
% else
%     confounder_vars = true;
% end

[V, ~, S] = size(B);

%%%% selection performance: for whole tensor B (V, V, S) %%%%%
mask_v = triu(true(V, V), 1); % (V, V)
B_triu = reshape(B(repmat(mask_v,1,1,S)), [], S); % (V2, S)
B_triu_nonzero = B_triu ~= 0; % (V2, S)
B_triu_nonzero_vtx = mean(B_triu_nonzero, 1); % (1, S)
B_hat_triu = reshape(B_hat(repmat(mask_v,1,1,S)), [], S); % (V2, S)
B_hat_triu_nonzero = B_hat_triu ~= 0; % (V2, S)
B_hat_triu_nonzero_vtx = mean(B_hat_triu_nonzero, 1); % (1, S)
B_triu_nonzero_vtx_binary = B_triu_nonzero_vtx > 0; % (1, S)
B_hat_triu_nonzero_vtx_binary = B_hat_triu_nonzero_vtx > 0; % (1, S)
% for every node
% V2 = size(B_triu, 1);
AUC_B_nodes = nan;
% AUC_B_nodes = zeros(1, V2);
% for vv = 1:V2
%     [~, ~, ~, AUC_B_nodes(vv)] = perfcurve(B_triu_nonzero(vv, :), B_hat_triu(vv, :), 1);
% end

% at least one nonzero element for each vtx
[~, ~, ~, AUC_B_vtx] = perfcurve(B_triu_nonzero_vtx_binary, B_hat_triu_nonzero_vtx, 1); % scalar
C_B_vtx = confusionmat(B_triu_nonzero_vtx_binary, B_hat_triu_nonzero_vtx_binary, 'order', [0,1]);
TP = C_B_vtx(2, 2, :);
TN = C_B_vtx(1, 1, :);
FP = C_B_vtx(1, 2, :);
FN = C_B_vtx(2, 1, :);
TPR_B_vtx = squeeze(TP ./ (TP + FN));
TDR_B_vtx = squeeze(TP ./ (TP + FP));
FDR_B_vtx = squeeze(FP ./ (FP + TP));
FPR_B_vtx = squeeze(FP ./ (FP + TN));
Accuracy_B_vtx = mean(B_triu_nonzero_vtx_binary == B_hat_triu_nonzero_vtx_binary); % scalar
F1_B_vtx = squeeze(2 * TP ./ (2 * TP + FP + FN));
df_B_vtx_indicator = table(TPR_B_vtx, TDR_B_vtx, FDR_B_vtx, FPR_B_vtx, Accuracy_B_vtx, F1_B_vtx, AUC_B_vtx);

% OVERALL
[~, ~, ~, AUC_B] = perfcurve(B_triu_nonzero(:), B_hat_triu(:), 1); % scalar
C_B = confusionmat(B_triu_nonzero(:), B_hat_triu_nonzero(:), 'order', [0,1]);
TP = C_B(2, 2, :);
TN = C_B(1, 1, :);
FP = C_B(1, 2, :);
FN = C_B(2, 1, :);
TPR_B = squeeze(TP ./ (TP + FN));
TDR_B = squeeze(TP ./ (TP + FP));
FDR_B = squeeze(FP ./ (FP + TP));
FPR_B = squeeze(FP ./ (FP + TN));
Accuracy_B = mean(B_triu_nonzero == B_hat_triu_nonzero, "all"); % (V2, 1)
F1_B = squeeze(2 * TP ./ (2 * TP + FP + FN));
df_B_indicator = table(TPR_B, TDR_B, FDR_B, FPR_B, Accuracy_B, F1_B, AUC_B);


%%%% estimation errors: for whole tensor B %%%%%%%
B_diff_abs = abs(B_triu - B_hat_triu);
MAE_B = mean(B_diff_abs, "all"); % scalar
MSE_B = mean(B_diff_abs.^2, "all"); % scalar
MAE_B_nonzero = mean(B_diff_abs(B_triu_nonzero), "all");
MSE_B_nonzero = mean(B_diff_abs(B_triu_nonzero).^2, "all");
MAE_B_zero = mean(B_diff_abs(~B_triu_nonzero), "all");
MSE_B_zero = mean(B_diff_abs(~B_triu_nonzero).^2, "all");
B_errors = table(MAE_B, MSE_B, MAE_B_nonzero, MSE_B_nonzero, MAE_B_zero, MSE_B_zero);


% %% R2 & MSE for response & BM
% BM_hat = permute(pagemtimes(M, permute(B_hat, [3, 1, 2])), [2, 3, 1]); % (V, V, n)
% if confounder_vars
%     Gamma_x_hat = permute(pagemtimes(x, permute(Gamma_hat, [3, 1, 2])), [2, 3, 1]); % (V, V, n)
%     A_hat = Gamma_x_hat + BM_hat;
% else
%     A_hat = BM_hat;
% end
% n = size(A, 3);
% mask_v = repmat(triu(true(V), 1), 1, 1, n);
% A_hat_triu = reshape(A_hat(mask_v), [], n); % (V2, n)
% A_triu = reshape(A(mask_v), [], n); % (V2, n)
% BM_triu = reshape(BM(mask_v), [], n); % (V2, n)
% BM_hat_triu = reshape(BM_hat(mask_v), [], n); % (V2, n)
% R2_hat_all = zeros(1, n);
% R2_hat_BM_all = zeros(1, n);
% for ii = 1:n
%     R2_hat_all(ii) = corr(A_triu(:,ii), A_hat_triu(:,ii))^2;
%     R2_hat_BM_all(ii) = corr(BM_triu(:,ii), BM_hat_triu(:,ii))^2;
% end
% R2_hat = mean(R2_hat_all);
% R2_hat_BM = mean(R2_hat_BM_all);
% A_triu_diff_abs = abs(A_triu - A_hat_triu);
% MAE_A_hat = mean(A_triu_diff_abs, 'all');
% MSE_A_hat = mean(A_triu_diff_abs.^2, 'all');
% BM_triu_diff_abs = abs(BM_triu - BM_hat_triu);
% MAE_BM_hat = mean(BM_triu_diff_abs, 'all');
% MSE_BM_hat = mean(BM_triu_diff_abs.^2, 'all');

% fitted_errors = table(R2_hat, MAE_A_hat, MSE_A_hat, R2_hat_BM, MAE_BM_hat, MSE_BM_hat);


end