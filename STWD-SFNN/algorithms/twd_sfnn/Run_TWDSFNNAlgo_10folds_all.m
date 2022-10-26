clear
clc
rng(0)
warning off
set(0,'DefaultFigureVisible', 'off')

file_path_read = 'F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';  % 读取文件路径      
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU', ...
             'DCC', 'SB', 'EOL', 'BM', 'ESR', ...
             'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % 处理的文件名称
file_name_per = file_name(5);
load(['C:\Users\Lenovo\Desktop\data_is_0911\'  char(file_name_per)  '.mat'])
file_path_save = 'F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\Result_algorithms\10folds_PSO_0922\'; % 读取文件路径
mkdir(file_path_save) 
Para_Init.data_slide = 1;     % 需要划分测试集
Para_Init.Data_Type = 2;      % 部分归一化处理

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folds_A = 10;
Para_Init.file_path_read = file_path_read;
Para_Init.file_name_per = file_name_per;
Para_Init.file_path_save = file_path_save;
Para_Init.folds_A = folds_A;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 加载预存的 twd %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([file_path_read  char(file_name_per)  '_TWDSFNN_para_twd'  '.mat'])
Para_Init.TWD_Threshold_init = threshold_init;
Para_Init.TWD_lambda_pp = lambda_matrix(1,1);
Para_Init.TWD_lambda_bp = lambda_matrix(1,2);
Para_Init.TWD_lambda_np = lambda_matrix(1,3);
Para_Init.TWD_lambda_nn = lambda_matrix(2,1);
Para_Init.TWD_lambda_bn = lambda_matrix(2,2);
Para_Init.TWD_lambda_pn = lambda_matrix(2,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 加载预存的 10cv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([file_path_read  char(file_name_per)  '_TWDSFNN_'  num2str(folds_A)  'cv'  '.mat']) 
indices = indices_10cv;

active_list = [1, 3, 7, 8, 10, 11];
data_type = [1, 2];
Result_matrix_all = [];
runtimes = 1;
for i = 1:length(active_list)
    Para_Init.s = active_list(i);   % 激活函数类型,前7个是Relu函数,后4个是tanh函数
    for j = 1:length(data_type)
        Para_Init.t = data_type(j);   % 数据的分布类型,1是服从均匀分布,否则服从正态分布
        disp('***********************************************************')
        fprintf('激活函数类型=%d\n',Para_Init.s)
        fprintf('数据的分布类型=%d\n',Para_Init.t)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 加载预存的 wb %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        load([file_path_read  char(file_name_per)  '_SFNN_para_'  num2str(Para_Init.s) num2str(Para_Init.t) '.mat'])
        Para_Init.Weight_1 = Weight_1;
        Para_Init.bias_1   = bias_1;
        Para_Init.Weight_2 = Weight_2;
        Para_Init.bias_2   = bias_2;
        
        [Result_TraiNums, Result_matrix, Result_Mean,Result_SE] = Run_TWDSFNNAlgo_10folds(data, label, Para_Init, indices);
        Result_train_all(runtimes,:) = Result_TraiNums;  % 存储一组参数下stwd-sfnn在每个阶段的训练样本数目
        Result_matrix_all{runtimes} = [Result_matrix; Result_Mean; Result_SE];
        runtimes = runtimes + 1;
        disp('***********************************************************')
    end
end
% 保存实验结果
save([file_path_save  char(file_name_per)  '_TWDSFNN_10folds_all'  '.mat'], 'Result_matrix_all','Result_train_all')
        
