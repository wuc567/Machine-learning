function [Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_Result)
Contrast_Result_A_Mean_all = [];Contrast_Result_A_SE_all = [];
Contrast_Result_A_Mean = [];  % 所有参数下的样本的 Mean
Contrast_Result_A_SE   = [];  % 所有参数下的样本的 SE
[folds,para_nums] = size(Contrast_Result);
for j = 1: para_nums
    Contrast_Result_para_nums_per = Contrast_Result(:,j); % 第 j 组参数下的10折实验结果, 10*1 cell
    Contrast_Result_A_temp = []; 
    
    for k = 1: folds
        Contrast_Result_para_folds_per = Contrast_Result_para_nums_per{k}; % 第 j 组参数下的第 k 折实验结果, 1*17 double
        [Results_A, Results_B] = Results_split(Contrast_Result_para_folds_per);
        Contrast_Result_A_temp = [Contrast_Result_A_temp; Results_A];
        Contrast_Result_B_temp{j,k} = Results_B;
    end
    
    % 计算所有参数下的样本的 Mean 和 SE
    Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % 第 j 组参数下的样本的 Mean
    Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % 第 j 组参数下的样本的 SE(95%的置信区间)
    Contrast_Result_A_Mean      = [Contrast_Result_A_Mean; Contrast_Result_A_temp_Mean]; % 所有参数下的样本的 Mean
    Contrast_Result_A_SE        = [Contrast_Result_A_SE;   Contrast_Result_A_temp_SE];   % 所有参数下的样本的 SE
end

% 返回最好的 Mean 和 SE
% Test_Acc(max), Test_WeightF1(max), Test_Kappa(max), Test_Loss(min),Para_Init.Hidden_step(min), Traintime, Testtime
[Contrast_Result_A_Mean_sort_list,Contrast_Result_A_Mean_sort_index] = sortrows(Contrast_Result_A_Mean, [-1,-1,-1,1,1,1,1]); % -1是倒序排列
Contrast_Result_A_Mean_i       = Contrast_Result_A_Mean_sort_list(1,:); % 取排序后的首行
Contrast_Result_A_Mean_i_index = Contrast_Result_A_Mean_sort_index(1);  % 取排序后的首行的位置索引
Contrast_Result_A_SE_i         = Contrast_Result_A_SE(Contrast_Result_A_Mean_i_index,:); % 取排序后的首行

Contrast_Result_A_Mean_all = [Contrast_Result_A_Mean_all; Contrast_Result_A_Mean_i];
Contrast_Result_A_SE_all   = [Contrast_Result_A_SE_all;   Contrast_Result_A_SE_i];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Results_A, Results_C] = Results_split(Results_vectors)
Results_11111 = find(Results_vectors == 11111);
Results_22222 = find(Results_vectors == 22222);
d = Results_22222 - Results_11111 - 1;

Results_33333 = find(Results_vectors == 33333);
e = Results_33333 + d ;

if e == length(Results_vectors)
    Results_A = Results_vectors(1:Results_11111-1); % Test_Acc, Test_WeightF1, Test_Kappa, Test_Loss, Para_Init.Hidden_step
    Results_B = Results_vectors(Results_11111:end); % 111111, Cost_Result, 222222, Cost_Test, 333333,Cost_Delay
    Results_C = reshape(Results_B,length(Results_B)/3, 3)'; % 111111, Cost_Result; 222222, Cost_Test; 333333,Cost_Delay
else
    Results_A = Results_vectors;%(:,1:7);
    Results_C = [];
    disp('Spectial case !!!')
end
end