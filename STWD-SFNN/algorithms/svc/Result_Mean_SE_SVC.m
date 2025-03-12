function [Contrast_Result_A_Mean,Contrast_Result_A_SE] = Result_Mean_SE_SVC(Contrast_Result)

Contrast_Result_A_Mean = []; Contrast_Result_A_SE = [];
[folds,para_nums] = size(Contrast_Result);
for j = 1: para_nums
    Contrast_Result_A_temp = Contrast_Result(:,j); % �� j ������µ�10��ʵ����, 10*1 cell
    
    % �������в����µ������� Mean �� SE
    Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % �� j ������µ������� Mean
    Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % �� j ������µ������� SE(95%����������)
    Contrast_Result_A_Mean      = [Contrast_Result_A_Mean, Contrast_Result_A_temp_Mean]; % ���в����µ������� Mean
    Contrast_Result_A_SE        = [Contrast_Result_A_SE,   Contrast_Result_A_temp_SE];   % ���в����µ������� SE
end
end
