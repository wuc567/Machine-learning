clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

file_path_read = 'F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';  % 保存文件路径       
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU',...
             'DCC', 'SB', 'EOL', 'BM', 'ESR',...
             'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % 处理的文件名称
file_name_per = file_name(5);
load(['C:\Users\Lenovo\Desktop\data_is_0911\'  char(file_name_per)  '.mat'])

Para_Init.data_slide = 1;    % 需要划分测试集
Para_Init.Data_Type = 2;     % 部分归一化处理
folds_A = 10;
file_path_save = 'F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\Result_algorithms\10folds_PSO_0922\'; % 读取文件路径
mkdir(file_path_save) 
Para_Init.file_path_read = file_path_read;
Para_Init.file_name_per = file_name_per;
Para_Init.file_path_save = file_path_save;
Para_Init.folds_A = folds_A;

Para_Init.data_r = size(data,1); % 数据集的样本量
Para_Init.data_c = size(data,2); % 数据集的特征数
Para_Init.ClassNum = numel(unique(label));  % 数据集的类别数目
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

%对比算法之隐层节点数目
Para_Init.eta = 2; %介于1~10之间
K1 = ceil(sqrt(Para_Init.data_c + Para_Init.ClassNum+Para_Init.eta));
K2 = ceil(log2(Para_Init.data_c));
K3 = ceil(sqrt(Para_Init.data_c * Para_Init.ClassNum));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 加载预存的 10cv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([file_path_read  char(file_name_per)  '_TWDSFNN_'  num2str(folds_A)  'cv'  '.mat']) 
Para_Init.indices = indices_10cv;

iter = 1;
Result_K123 = [];
for k = [K1,K2,K3]
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

            [Result_matrix, Result_Mean, Result_SE] = Para_Contrast_Hidden_K123(data,label,label_onehot,Para_Init,k,iter);
            Result_matrix_all{runtimes} = [Result_matrix; Result_Mean; Result_SE];
            runtimes = runtimes + 1;
            disp('***********************************************************')
        end
    end 
    Result_K123{iter} = Result_matrix_all;
    iter = iter + 1; 
    fprintf('##### 对比算法 ##### = %d\n',iter)  
end
save([Para_Init.file_path_save  char(Para_Init.file_name_per)  '_SFNN_K123'  '.mat'], 'Result_K123')
