clc
clear

% STWD-SFNN�㷨
folder = ('E:\PaperTwo\Version-revise\Revise-1010\26��STWDNN\mean_se\');
file_list_all = dir([folder,'*.mat']);
file_list = [];
for i = 1:length(file_list_all)
   file_list{i,1} = file_list_all(i).name; % ֻȡ name ��
end
clear i  file_list_all   

% ����ÿ�����ݼ��µ�mat�ļ�
i = 1;
Contrast_Result_A_Mean_all = []; 
Contrast_Result_A_SE_all = [];   
Contrast_Result_B_index  = [];

while i <= 12
    file_list_dataset_per       = file_list{i};
    file_list_dataset_per_split = strsplit(file_list_dataset_per,'_'); % ���ա�_���ַ��ֿ�
    
    
    Result_Para_list    = load([folder, file_list_dataset_per]) ;  % �� BM_STWDSFNN_11.mat
    Contrast_Result_per = Result_Para_list.Contrast_SFNN_Result;   % �� ��Ҫ�ı���
    
    Contrast_Result_A_Mean = [];  % ���в����µ������� Mean
    Contrast_Result_A_SE   = [];  % ���в����µ������� SE
    [folds,para_nums] = size(Contrast_Result_per);
    for j = 1: para_nums
        Contrast_Result_para_nums_per = Contrast_Result_per(:,j); % �� j ������µ�10��ʵ����, 10*1 cell
        Contrast_Result_A_temp = []; 
        
        for k = 1: folds
            Contrast_Result_para_folds_per = Contrast_Result_para_nums_per{k}; % �� j ������µĵ� k ��ʵ����, 1*17 double
            [Results_A, Results_B] = Results_split(Contrast_Result_para_folds_per);
            Contrast_Result_A_temp = [Contrast_Result_A_temp; Results_A];
            Contrast_Result_B_temp{j,k} = Results_B;
        end  
        
        % �������в����µ������� Mean �� SE
        Contrast_Result_A_temp_Mean = mean(Contrast_Result_A_temp);                % �� j ������µ������� Mean
        Contrast_Result_A_temp_SE   = 2 * std(Contrast_Result_A_temp)/sqrt(folds); % �� j ������µ������� SE(95%����������)
        Contrast_Result_A_Mean      = [Contrast_Result_A_Mean; Contrast_Result_A_temp_Mean]; % ���в����µ������� Mean
        Contrast_Result_A_SE        = [Contrast_Result_A_SE;   Contrast_Result_A_temp_SE];   % ���в����µ������� SE
    end
    
    % ������õ� Mean �� SE
    % Test_Acc(max), Test_WeightF1(max), Test_Kappa(max), Test_Loss(min),
    % Para_Init.Hidden_step(min), Traintime, Testtime
    [Contrast_Result_A_Mean_sort_list,Contrast_Result_A_Mean_sort_index] = sortrows(Contrast_Result_A_Mean, [-1,-1,-1,1,1,1,1]); % -1�ǵ�������
    Contrast_Result_A_Mean_i       = Contrast_Result_A_Mean_sort_list(1,:); % ȡ����������
    Contrast_Result_A_Mean_i_index = Contrast_Result_A_Mean_sort_index(1);  % ȡ���������е�λ������
    Contrast_Result_A_SE_i         = Contrast_Result_A_SE(Contrast_Result_A_Mean_i_index,:); % ȡ����������
    
    Contrast_Result_A_Mean_all = [Contrast_Result_A_Mean_all; Contrast_Result_A_Mean_i];
    Contrast_Result_A_SE_all   = [Contrast_Result_A_SE_all;   Contrast_Result_A_SE_i];
    Contrast_Result_B_index    = [Contrast_Result_B_index; [i,Contrast_Result_A_Mean_i_index]];
    
    i = i + 1;
end
[Contrast_Result_A_Mean_all_sort_list,Contrast_Result_A_Mean_all_sort_index] = sortrows(Contrast_Result_A_Mean_all, [-1,-1,-1,1,1]); % -1�ǵ�������
Contrast_Result_A_Mean_best       = Contrast_Result_A_Mean_all_sort_list(1,:); % ȡ����������
Contrast_Result_A_Mean_best_index = Contrast_Result_A_Mean_all_sort_index(1);  % ȡ���������е�λ������
Contrast_Result_A_SE_best         = Contrast_Result_A_SE_all(Contrast_Result_A_Mean_best_index,:); % ȡ����������

 



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