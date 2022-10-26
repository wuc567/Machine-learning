function [Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_Result)
Contrast_Result = Contrast_Result'; % fold * ��������Ŀ
Contrast_Result_A_Mean_all = [];Contrast_Result_A_SE_all = [];
Contrast_Result_A_Mean = [];  % ���в����µ������� Mean
Contrast_Result_A_SE   = [];  % ���в����µ������� SE
[folds,para_nums] = size(Contrast_Result);
for j = 1: para_nums
    Contrast_Result_para_nums_per = Contrast_Result(:,j); % �� j ������µ�10��ʵ����, 10*1 cell
    Contrast_Result_A_temp = []; 
    
    for k = 1: folds
        Contrast_Result_para_folds_per = Contrast_Result_para_nums_per{k}; % �� j ������µĵ� k ��ʵ����, 1*17 double
        Contrast_Result_A_temp = [Contrast_Result_A_temp; Contrast_Result_para_folds_per];
    end
    
    % �������в����µ������� Mean �� SE
    Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % �� j ������µ������� Mean
    Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % �� j ������µ������� SE(95%����������)
    Contrast_Result_A_Mean      = [Contrast_Result_A_Mean; Contrast_Result_A_temp_Mean]; % ���в����µ������� Mean
    Contrast_Result_A_SE        = [Contrast_Result_A_SE;   Contrast_Result_A_temp_SE];   % ���в����µ������� SE
end

% ������õ� Mean �� SE
% Test_WeightF1(max), Test_Acc(max), Traintime, Testtime
[Contrast_Result_A_Mean_sort_list,Contrast_Result_A_Mean_sort_index] = sortrows(Contrast_Result_A_Mean, [-1,-1,1,1]); % -1�ǵ�������
Contrast_Result_A_Mean_i       = Contrast_Result_A_Mean_sort_list(1,:); % ȡ����������
Contrast_Result_A_Mean_i_index = Contrast_Result_A_Mean_sort_index(1);  % ȡ���������е�λ������
Contrast_Result_A_SE_i         = Contrast_Result_A_SE(Contrast_Result_A_Mean_i_index,:); % ȡ����������

Contrast_Result_A_Mean_all = [Contrast_Result_A_Mean_all; Contrast_Result_A_Mean_i];
Contrast_Result_A_SE_all   = [Contrast_Result_A_SE_all;   Contrast_Result_A_SE_i];
end
