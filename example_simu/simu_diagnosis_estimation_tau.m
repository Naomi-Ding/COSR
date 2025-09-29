%% Simulation errors
function [alpha_hat, alpha_hat_thresholded, tau_hat, ...
    Z_df, df_tau_indicator, tau_errors] = ...
    simu_diagnosis_estimation_tau(tau, Z, H_new, alpha_hat, ...
    alpha_hat_thresholded, tau_hat, Z_hat_idx)
% Z_hat: (V, H_new)
% alpha_hat: (H2_new, S)
% alpha_hat_thresholded: (H2_new, S)
% tau_hat: (H2_new, S)
% B_hat: (V, V, S)
% Z_df: (1, 6) table
% df_tau_indicator: (H2_new, 7) table or empty
% tau_errors: (1, 6) table or empty
% AUC_B_nodes: (1, V2)
% df_B_vtx_indicator: (1, 7) table
% df_B_indicator: (1, 7) table
% B_errors: (1, 6) table
% fitted_errors: (1, 6) table

[V, H] = size(Z);
% S = size(M, 2);
H2 = H * (H + 1) / 2;
H2_new = H_new * (H_new + 1) /2;
mask_h_tri = triu(true(H_new));
mask_h_triu = triu(true(H_new), 1);
mask_h_tril = tril(true(H_new), -1);


%%%%%%%%%% (1) membership %%%%%%%%%%
[~, Z_idx] = max(Z == 1, [], 2); % (V, 1)
% [~, Z_hat_idx] = max(Z_hat == 1, [], 2); % (V, 1)
% [~, Z_hat_idx] = max(E_Z_hat, [], 2); % (V, 1)
% Z_hat = double(bsxfun(@eq, (1:H_new), Z_hat_idx)); % (V, H_new)

[RI, ARI, TP, TN, FP, FN] = randindex(Z_hat_idx, Z_idx);
TPR_Z = TP ./ (TP + FN);
TDR_Z = TP ./ (TP + FP);
FPR_Z = FP ./ (FP + TN);
FDR_Z = FP ./ (FP + TP);
Z_df = table(RI, ARI, TPR_Z, TDR_Z, FPR_Z, FDR_Z, 'VariableNames', ...
    ["RI", "ARI", "TPR", "TDR", "FPR", "FDR"]);

if H2 == H2_new
    % Create a mapping from Z_hat clusters to Z clusters
    Z_idx = grp2idx(Z_idx);
    Z_hat_idx = grp2idx(Z_hat_idx);
    ConfMat = zeros(H, H_new);
    for i = 1:V
        ConfMat(Z_idx(i), Z_hat_idx(i)) = ConfMat(Z_idx(i), Z_hat_idx(i)) + 1;
    end
    % Solve the assignment problem (Hungarian algorithm)
    [assignment, ~] = munkres(-ConfMat);  % Min-cost matching (flip sign for max matching)
    [Z_hat_to_Z_map, ~] = find(assignment');

    % [unique_pairs, ~, ~] = unique([Z_idx(:), Z_hat_idx(:)], 'rows');
    % Z_hat_to_Z_map = zeros(H_new, 1);
    % for i = 1:size(unique_pairs, 1)
    %     Z_hat_to_Z_map(unique_pairs(i, 1)) = unique_pairs(i, 2);
    % end
    % % disp(Z_hat_to_Z_map)
    % Z_hat_zero = Z_hat_to_Z_map == 0;
    % if any(Z_hat_zero)
    %     Z_hat_to_Z_map(Z_hat_zero) = 1;
    % end

    Z_correspondence = zeros(H_new, H_new);
    Z_correspondence(mask_h_tri) = 1:H2_new;
    tmp = Z_correspondence';
    Z_correspondence(mask_h_tril) = tmp(mask_h_tril);
    Z_correspondence = Z_correspondence(Z_hat_to_Z_map, Z_hat_to_Z_map);
    mask_h_tri2 = triu(true(H));
    Z_correspondence = Z_correspondence(mask_h_tri2)'; % (1, H2)


    %%%%%%%%%% (2) community-wise coefficients %%%%%%%%%%
    tau_hat = tau_hat(Z_correspondence, :); % (H2, S)
    alpha_hat = alpha_hat(Z_correspondence, :); % (H2, S)
    % alpha_hat_thresholded = abs(alpha_hat) > delta_hat; % (H2, S)
    alpha_hat_thresholded = alpha_hat_thresholded(Z_correspondence, :);  % (H2, S)


    %%%% selection performance: for each tau(s) %%%%
    tau_indicator = tau ~= 0; % (H2, S)
    C = zeros(2,2,H2);
    AUC = NaN(H2, 1);
    for k = 1:H2
        % confusion matrix
        C(:,:,k) = confusionmat(tau_indicator(k,:), alpha_hat_thresholded(k, :), 'order', [0,1]);
        % AUC-ROC
        if numel(unique(tau_indicator(k,:))) == 2
            [~, ~, ~, AUC(k)] = perfcurve(tau_indicator(k,:), abs(alpha_hat(k,:)), 1);
        end
    end
    TP = C(2, 2, :);
    TN = C(1, 1, :);
    FP = C(1, 2, :);
    FN = C(2, 1, :);
    TPR = squeeze(TP ./ (TP + FN));
    TDR = squeeze(TP ./ (TP + FP));
    FDR = squeeze(FP ./ (FP + TP));
    FPR = squeeze(FP ./ (FP + TN));
    Accuracy = mean(tau_indicator == alpha_hat_thresholded, 2); % (H2, 1)
    F1 = squeeze(2 * TP ./ (2 * TP + FP + FN));
    df_tau_indicator = table(TPR, TDR, FPR, FDR, Accuracy, F1, AUC);
    % disp(df_tau_indicator)

    % estimation errors
    tau_diff_abs = abs(tau - tau_hat);
    MAE_tau = mean(tau_diff_abs, 2)'; % (1, H2_new)
    MSE_tau = mean(tau_diff_abs.^2, 2)'; % (1, H2_new)
    MAE_tau_nonzero = mean(tau_diff_abs(tau_indicator), "all"); % scalar
    MSE_tau_nonzero = mean(tau_diff_abs(tau_indicator).^2, "all"); % scalar
    MAE_tau_zero = mean(tau_diff_abs(~tau_indicator), "all"); % scalar
    MSE_tau_zero = mean(tau_diff_abs(~tau_indicator).^2, "all"); % scalar
    tau_errors = table(MAE_tau, MSE_tau, MAE_tau_nonzero, MSE_tau_nonzero, MAE_tau_zero, MSE_tau_zero);
else
    df_tau_indicator = [];
    tau_errors = [];
end



end