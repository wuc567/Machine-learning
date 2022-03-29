function [Contrast_Result_A_Mean,Contrast_Result_A_SE] = Result_Mean_SE(Contrast_Result)
Contrast_Result_matrix = [];
folds = size(Contrast_Result,2);
for i = 1: folds
    Contrast_Result_matrix = [Contrast_Result_matrix; Contrast_Result{i}];
end

% �������в����µ������� Mean �� SE
Contrast_Result_A_Mean = mean(Contrast_Result_matrix);                % �� j ������µ������� Mean
Contrast_Result_A_SE   = 2 * std(Contrast_Result_matrix)/sqrt(folds); % �� j ������µ������� SE(95%����������)
