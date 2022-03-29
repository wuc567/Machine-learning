function [Contrast_Result_A_Mean,Contrast_Result_A_SE] = Result_Mean_SE(Contrast_Result)
Contrast_Result_matrix = [];
folds = size(Contrast_Result,2);
for i = 1: folds
    Contrast_Result_matrix = [Contrast_Result_matrix; Contrast_Result{i}];
end

% 计算所有参数下的样本的 Mean 和 SE
Contrast_Result_A_Mean = mean(Contrast_Result_matrix);                % 第 j 组参数下的样本的 Mean
Contrast_Result_A_SE   = 2 * std(Contrast_Result_matrix)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)
