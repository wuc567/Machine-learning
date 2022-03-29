clc
clear

% STWD-SFNN算法
folder = ('E:\PaperTwo\Version-revise\Revise-1010\26—STWDNN\mean_se\');
file_list_all = dir([folder,'*.mat']);
file_list = [];
for i = 1:length(file_list_all)
   file_list{i,1} = file_list_all(i).name; % 只取 name 列
end
clear i  file_list_all   

% 遍历每个数据集下的mat文件
i = 1;
Contrast_Result_A_Mean_all = []; 
Contrast_Result_A_SE_all = [];   
Contrast_Result_B_index  = [];

while i <= 12
    file_list_dataset_per       = file_list{i};
    file_list_dataset_per_split = strsplit(file_list_dataset_per,'_'); % 按照‘_’字符分开
    
    
    Result_Para_list    = load([folder, file_list_dataset_per]) ;  % 打开 BM_STWDSFNN_11.mat
    Contrast_Result_per = Result_Para_list.Contrast_SFNN_Result;   % 打开 需要的变量
    
    Contrast_Result_A_Mean = [];  % 所有参数下的样本的 Mean
    Contrast_Result_A_SE   = [];  % 所有参数下的样本的 SE
    [folds,para_nums] = size(Contrast_Result_per);
    for j = 1: para_nums
        Contrast_Result_para_nums_per = Contrast_Result_per(:,j); % 第 j 组参数下的10折实验结果, 10*1 cell
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
    % Test_Acc(max), Test_WeightF1(max), Test_Kappa(max), Test_Loss(min),
    % Para_Init.Hidden_step(min), Traintime, Testtime
    [Contrast_Result_A_Mean_sort_list,Contrast_Result_A_Mean_sort_index] = sortrows(Contrast_Result_A_Mean, [-1,-1,-1,1,1,1,1]); % -1是倒序排列
    Contrast_Result_A_Mean_i       = Contrast_Result_A_Mean_sort_list(1,:); % 取排序后的首行
    Contrast_Result_A_Mean_i_index = Contrast_Result_A_Mean_sort_index(1);  % 取排序后的首行的位置索引
    Contrast_Result_A_SE_i         = Contrast_Result_A_SE(Contrast_Result_A_Mean_i_index,:); % 取排序后的首行
    
    Contrast_Result_A_Mean_all = [Contrast_Result_A_Mean_all; Contrast_Result_A_Mean_i];
    Contrast_Result_A_SE_all   = [Contrast_Result_A_SE_all;   Contrast_Result_A_SE_i];
    Contrast_Result_B_index    = [Contrast_Result_B_index; [i,Contrast_Result_A_Mean_i_index]];
    
    i = i + 1;
end
[Contrast_Result_A_Mean_all_sort_list,Contrast_Result_A_Mean_all_sort_index] = sortrows(Contrast_Result_A_Mean_all, [-1,-1,-1,1,1]); % -1是倒序排列
Contrast_Result_A_Mean_best       = Contrast_Result_A_Mean_all_sort_list(1,:); % 取排序后的首行
Contrast_Result_A_Mean_best_index = Contrast_Result_A_Mean_all_sort_index(1);  % 取排序后的首行的位置索引
Contrast_Result_A_SE_best         = Contrast_Result_A_SE_all(Contrast_Result_A_Mean_best_index,:); % 取排序后的首行

 



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
    disp('Spectial case !!!')
end
end