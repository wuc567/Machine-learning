function [Contrast_Result_A_Mean, Contrast_Result_A_SE, Contrast_Result_K_Mean, Contrast_Result_K_SE, Contrast_Result_optimal] = Result_Mean_SE(Contrast_Result)
Contrast_Result_matrix = [];
folds = size(Contrast_Result,1);
for i = 1: folds
    Contrast_Result_matrix = [Contrast_Result_matrix; Contrast_Result{i,1}];
end
Contrast_Result_matrix_acc = Contrast_Result_matrix(:,1);  % ACC
[~, Contrast_Result_matrix_acc_index] = max(Contrast_Result_matrix_acc);
Contrast_Result_optimal= Contrast_Result(Contrast_Result_matrix_acc_index,:);

% 计算所有参数下的 Acc 样本的 Mean 和 SE
Contrast_Result_A_Mean = mean(Contrast_Result_matrix_acc);                % 第 j 组参数下的样本的 Mean
Contrast_Result_A_SE   = 2 * std(Contrast_Result_matrix_acc)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)

% 计算所有参数下的 K 值样本的 Mean 和 SE
Contrast_Result_matrix_K = Contrast_Result_matrix(:,8);  % K
Contrast_Result_K_Mean = mean(Contrast_Result_matrix_K);                % 第 j 组参数下的样本的 Mean
Contrast_Result_K_SE   = 2 * std(Contrast_Result_matrix_K)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)
