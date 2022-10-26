clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

file_path_read = 'E:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';  % 保存文件路径       
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU',...
             'DCC', 'SB', 'EOL', 'BM', 'ESR',...
             'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % 处理的文件名称
file_name_per = file_name(5);
load(['C:\Users\Lenovo\Desktop\data_is_0911\'  char(file_name_per)  '.mat'])
file_path_save = 'E:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\Result_algorithms\10folds_GS_0922\'; % 读取文件路径
mkdir(file_path_save) 
Para_Init.data_slide = 1;    % 需要划分测试集
Para_Init.Data_Type = 2;     % 部分归一化处理
folds_A = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para_Init.TWD_cases = 1;     % 讨论 TWD 参数不同取值下的情形
Para_Init.file_path_read = file_path_read;
Para_Init.file_name_per = file_name_per;
Para_Init.file_path_save = file_path_save;
Para_Init.folds_A = folds_A;

% 构建需要优化的参数列表
alpha = 0.1; %[0.1,0.01];     % 梯度下降的学习率 ,0.01,0.001
BatchSize = 512; %[128,256]; % 随机梯度下降的批大小 ,128,256
lambda = 0.1; %[1];        % 损失函数的正则项系数 ,0.1,1,10
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % 参数列表,48*3

Para_Init.p = 1;   % 不同的调参方法,1是SGD+Momentum,2是Adam,3是不带修正项的AMSgrad
Para_Init.Acc_init = 0.99; % 初始化的准确率
Para_Init.Loss_init = 1;  % 初始化的损失值
Para_Init.LossFun = 2;   % 损失函数类型,1是CE交叉熵损失函数,2是FL聚焦损失函数
Para_Init.FL_Adjust = 2; % FL聚焦损失函数的调整因子
Para_Init.Batch_epochs = 200; % 批大小的迭代次数
Para_Init.data_r = size(data,1); % 数据集的样本量
Para_Init.data_c = size(data,2); % 数据集的特征数
Para_Init.ClassNum = numel(unique(label));  % 数据集的类别数目
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot
Para_Init.folds = 10;   % 数据的10折交叉验证
Para_Init.Hidden_nodes_max = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 加载预存的 10cv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([file_path_read  char(file_name_per)  '_TWDSFNN_'  num2str(folds_A)  'cv'  '.mat']) 
indices = indices_10cv;

active_list = [1, 3, 7, 8, 10, 11];   % 激活函数类型,前7个是Relu函数,后4个是tanh函数
data_type = [1, 2];                   % 数据的分布类型,1是服从均匀分布,否则服从正态分布
Result_matrix_all = [];
runtimes = 1;
for i = 1:length(active_list)
    Para_Init.s = active_list(i);     % 激活函数类型,前7个是Relu函数,后4个是tanh函数
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
        
        [Result_matrix, Result_Mean,Result_SE] = Run_GS_10folds(data, label, Para_Init, indices);
        Result_matrix_all{runtimes} = [Result_matrix; Result_Mean; Result_SE];
        runtimes = runtimes + 1;
        disp('***********************************************************')
    end
end
% 保存实验结果
save([file_path_save  char(file_name_per)  '_GS_SFNN_10folds_all'  '.mat'], 'Result_matrix_all')
        
