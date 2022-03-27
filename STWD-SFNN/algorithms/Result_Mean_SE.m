function [Contrast_Result_A_Mean, Contrast_Result_A_SE, Contrast_Result_K_Mean, Contrast_Result_K_SE, Contrast_Result_optimal] = Result_Mean_SE(Contrast_Result)
Contrast_Result_matrix = [];
folds = size(Contrast_Result,1);
for i = 1: folds
    Contrast_Result_matrix = [Contrast_Result_matrix; Contrast_Result{i,1}];
end
Contrast_Result_matrix_acc = Contrast_Result_matrix(:,1);  % ACC
[~, Contrast_Result_matrix_acc_index] = max(Contrast_Result_matrix_acc);
Contrast_Result_optimal= Contrast_Result(Contrast_Result_matrix_acc_index,:);

% �������в����µ� Acc ������ Mean �� SE
Contrast_Result_A_Mean = mean(Contrast_Result_matrix_acc);                % �� j ������µ������� Mean
Contrast_Result_A_SE   = 2 * std(Contrast_Result_matrix_acc)/sqrt(folds); % �� j ������µ������� SE(95%����������)

% �������в����µ� K ֵ������ Mean �� SE
Contrast_Result_matrix_K = Contrast_Result_matrix(:,8);  % K
Contrast_Result_K_Mean = mean(Contrast_Result_matrix_K);                % �� j ������µ������� Mean
Contrast_Result_K_SE   = 2 * std(Contrast_Result_matrix_K)/sqrt(folds); % �� j ������µ������� SE(95%����������)
