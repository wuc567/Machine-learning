function [Contrast_Result_A_Mean,Contrast_Result_A_SE] = Result_Mean_SE_SVC(Contrast_Result)

Contrast_Result_A_Mean = []; Contrast_Result_A_SE = [];
[folds,para_nums] = size(Contrast_Result);
for j = 1: para_nums
    Contrast_Result_A_temp = Contrast_Result(:,j); % 第 j 组参数下的10折实验结果, 10*1 cell
    
    % 计算所有参数下的样本的 Mean 和 SE
    Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % 第 j 组参数下的样本的 Mean
    Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)
    Contrast_Result_A_Mean      = [Contrast_Result_A_Mean, Contrast_Result_A_temp_Mean]; % 所有参数下的样本的 Mean
    Contrast_Result_A_SE        = [Contrast_Result_A_SE,   Contrast_Result_A_temp_SE];   % 所有参数下的样本的 SE
end
end
