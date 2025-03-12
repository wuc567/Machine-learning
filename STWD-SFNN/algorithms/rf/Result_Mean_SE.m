function [Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_Result)
Contrast_Result = Contrast_Result'; % fold * 参数组数目
Contrast_Result_A_Mean_all = [];Contrast_Result_A_SE_all = [];
Contrast_Result_A_Mean = [];  % 所有参数下的样本的 Mean
Contrast_Result_A_SE   = [];  % 所有参数下的样本的 SE
[folds,para_nums] = size(Contrast_Result);
for j = 1: para_nums
    Contrast_Result_para_nums_per = Contrast_Result(:,j); % 第 j 组参数下的10折实验结果, 10*1 cell
    Contrast_Result_A_temp = []; 
    
    for k = 1: folds
        Contrast_Result_para_folds_per = Contrast_Result_para_nums_per{k}; % 第 j 组参数下的第 k 折实验结果, 1*17 double
        Contrast_Result_A_temp = [Contrast_Result_A_temp; Contrast_Result_para_folds_per];
    end
    
    % 计算所有参数下的样本的 Mean 和 SE
    Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % 第 j 组参数下的样本的 Mean
    Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)
    Contrast_Result_A_Mean      = [Contrast_Result_A_Mean; Contrast_Result_A_temp_Mean]; % 所有参数下的样本的 Mean
    Contrast_Result_A_SE        = [Contrast_Result_A_SE;   Contrast_Result_A_temp_SE];   % 所有参数下的样本的 SE
end

% 返回最好的 Mean 和 SE
% Test_WeightF1(max), Test_Acc(max), Traintime, Testtime
[Contrast_Result_A_Mean_sort_list,Contrast_Result_A_Mean_sort_index] = sortrows(Contrast_Result_A_Mean, [-1,-1,1,1]); % -1是倒序排列
Contrast_Result_A_Mean_i       = Contrast_Result_A_Mean_sort_list(1,:); % 取排序后的首行
Contrast_Result_A_Mean_i_index = Contrast_Result_A_Mean_sort_index(1);  % 取排序后的首行的位置索引
Contrast_Result_A_SE_i         = Contrast_Result_A_SE(Contrast_Result_A_Mean_i_index,:); % 取排序后的首行

Contrast_Result_A_Mean_all = [Contrast_Result_A_Mean_all; Contrast_Result_A_Mean_i];
Contrast_Result_A_SE_all   = [Contrast_Result_A_SE_all;   Contrast_Result_A_SE_i];
end
