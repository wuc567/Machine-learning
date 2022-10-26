clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

file_path_read = 'E:\PaperTwo\220504��papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';  % �����ļ�·��       
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU',...
             'DCC', 'SB', 'EOL', 'BM', 'ESR',...
             'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % ������ļ�����
file_name_per = file_name(5);
load(['C:\Users\Lenovo\Desktop\data_is_0911\'  char(file_name_per)  '.mat'])
file_path_save = 'E:\PaperTwo\220504��papertwo-Chinese\Code-Chinese-two\UCI_file\Result_algorithms\10folds_GS_0922\'; % ��ȡ�ļ�·��
mkdir(file_path_save) 
Para_Init.data_slide = 1;    % ��Ҫ���ֲ��Լ�
Para_Init.Data_Type = 2;     % ���ֹ�һ������
folds_A = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para_Init.TWD_cases = 1;     % ���� TWD ������ͬȡֵ�µ�����
Para_Init.file_path_read = file_path_read;
Para_Init.file_name_per = file_name_per;
Para_Init.file_path_save = file_path_save;
Para_Init.folds_A = folds_A;

% ������Ҫ�Ż��Ĳ����б�
alpha = 0.1; %[0.1,0.01];     % �ݶ��½���ѧϰ�� ,0.01,0.001
BatchSize = 512; %[128,256]; % ����ݶ��½�������С ,128,256
lambda = 0.1; %[1];        % ��ʧ������������ϵ�� ,0.1,1,10
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % �����б�,48*3

Para_Init.p = 1;   % ��ͬ�ĵ��η���,1��SGD+Momentum,2��Adam,3�ǲ����������AMSgrad
Para_Init.Acc_init = 0.99; % ��ʼ����׼ȷ��
Para_Init.Loss_init = 1;  % ��ʼ������ʧֵ
Para_Init.LossFun = 2;   % ��ʧ��������,1��CE��������ʧ����,2��FL�۽���ʧ����
Para_Init.FL_Adjust = 2; % FL�۽���ʧ�����ĵ�������
Para_Init.Batch_epochs = 200; % ����С�ĵ�������
Para_Init.data_r = size(data,1); % ���ݼ���������
Para_Init.data_c = size(data,2); % ���ݼ���������
Para_Init.ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot
Para_Init.folds = 10;   % ���ݵ�10�۽�����֤
Para_Init.Hidden_nodes_max = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����Ԥ��� 10cv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([file_path_read  char(file_name_per)  '_TWDSFNN_'  num2str(folds_A)  'cv'  '.mat']) 
indices = indices_10cv;

active_list = [1, 3, 7, 8, 10, 11];   % ���������,ǰ7����Relu����,��4����tanh����
data_type = [1, 2];                   % ���ݵķֲ�����,1�Ƿ��Ӿ��ȷֲ�,���������̬�ֲ�
Result_matrix_all = [];
runtimes = 1;
for i = 1:length(active_list)
    Para_Init.s = active_list(i);     % ���������,ǰ7����Relu����,��4����tanh����
    for j = 1:length(data_type)
        Para_Init.t = data_type(j);   % ���ݵķֲ�����,1�Ƿ��Ӿ��ȷֲ�,���������̬�ֲ�
        disp('***********************************************************')
        fprintf('���������=%d\n',Para_Init.s)
        fprintf('���ݵķֲ�����=%d\n',Para_Init.t)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����Ԥ��� wb %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% ����ʵ����
save([file_path_save  char(file_name_per)  '_GS_SFNN_10folds_all'  '.mat'], 'Result_matrix_all')
        
