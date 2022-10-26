function [Test_Result, Test_Result_Mean,Test_Result_SE] = Run_GS_10folds(data, label, Para_Init, indices)
file_name_per = Para_Init.file_name_per;
file_path_save = Para_Init.file_path_save;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para_Init.TWD_cases = 2;     % 讨论 TWD 参数不同取值下的情形
Para_Init.data_r = size(data,1); % 数据集的样本量
Para_Init.data_c = size(data,2); % 数据集的特征数
Para_Init.ClassNum = numel(unique(label));  % 数据集的类别数目
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

% 构建需要优化的参数列表
% alpha =  [0.1,0.01,0.001];      % 梯度下降的学习率
% BatchSize = [512,1024];        % 随机梯度下降的批大小
% lambda = [0.1,1,10];         % 损失函数的正则项系数
alpha =  0.1; %[0.1,0.01,0.001];      % 梯度下降的学习率
BatchSize = 512; %[512,1024];        % 随机梯度下降的批大小
lambda = 0.1; %[0.1,1,10];         % 损失函数的正则项系数
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % 参数列表,48*3

% 初始化超参数
Para_Init.p = 1;   % 不同的调参方法,1是Adam,2是SGD+Momentum,3是不带修正项的AMSgrad
Para_Init.Acc_init = 0.9; % 初始化的准确率
Para_Init.Loss_init = 1;  % 初始化的损失值
Para_Init.LossFun = 2;    % 损失函数类型,1是CE交叉熵损失函数,2是FL聚焦损失函数
Para_Init.FL_Adjust = 2;  % FL聚焦损失函数的调整因子
Para_Init.Batch_epochs = 200; % 批大小的迭代次数
Para_Init.Data_epochs = 10;   % 数据的迭代次数
Para_Init.TWD_ClusterNum = 5;  % 离散化过程的簇类数目
Para_Init.TWD_sigma = 2;
Para_Init.TWD_lambda_pn = 0.5;  
Para_Init.Hidden_step = 1;   % 每次叠加一个
Para_Init.Train_num = 1;     % 隐层节点数目
clear TWD_threshold_pair TWD_alpha_init  TWD_beta_init TWD_gamma_init alpha BatchSize  alpha_1 BatchSize_1 lambda_1

% 交叉验证
Test_Result=[];
for k=1:10 %Para_Init.Data_epochs 
    fprintf('数据的交叉验证次数=%d\n',k)    
    
    % 划分数据集
    [Train,Validate,Test,Para_Init.slice_train] = Data_Partition(data,label,label_onehot,indices,k,Para_Init.Data_Type,Para_Init.data_slide);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GS-SFNN训练的隐藏层结点数目
    [Train_values, Train_Para,Para_Init] = GS_SFNN_Algorithm(Train,Validate,Para_Init,Para_Optimize);
    Train_Acc    = Train_values(1,1);
    Hidden_nodes = Train_values(1,5);
    Train_time   = Train_values(1,6); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PSO-SFNN测试最好的粒子参数下的网络学习性能
    tic
    [~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss] = SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                                                           Train_Para{1},Train_Para{2},Train_Para{3},Train_Para{4},...
                                                           Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    Test_time = toc;
    Test_Result_per = [Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss,Hidden_nodes,Train_time, Test_time];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 保存每折下的网络学习性能
    Test_Result = [Test_Result; Test_Result_per];
    fprintf('数据的交叉验证次数=%d\n',k)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MeanSE
[Test_Result_Mean,Test_Result_SE] = Result_MeanSE(Test_Result);

% 保存实验结果
save([file_path_save  char(file_name_per)  '_GS_SFNN_'  num2str(Para_Init.s) num2str(Para_Init.t)  '.mat'], ...
     'Test_Result','Test_Result_Mean', 'Test_Result_SE')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Contrast_Result_Mean,Contrast_Result_SE] = Result_MeanSE(Contrast_Result)
Contrast_Result_Mean = [];  % 所有参数下的样本的 Mean
Contrast_Result_SE   = [];  % 所有参数下的样本的 SE
[folds,columns] = size(Contrast_Result);
for j = 1: columns
    Contrast_Result_temp = Contrast_Result(:,j);  % 第 j 列
    
    % 计算所有参数下的样本的 Mean 和 SE
    Contrast_Result_temp_Mean = mean(Contrast_Result_temp);                        % 第 j 列的样本的 Mean
    Contrast_Result_temp_SE   = 2 * std(Contrast_Result_temp)/sqrt(folds);       % 第 j 列的样本的 SE(95%的置信区间)
    
    Contrast_Result_Mean      = [Contrast_Result_Mean, Contrast_Result_temp_Mean]; % 所有参数下的样本的 Mean
    Contrast_Result_SE        = [Contrast_Result_SE,   Contrast_Result_temp_SE];   % 所有参数下的样本的 SE
end
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Train,Validate,Test, slice_train] = Data_Partition(data,label,label_onehot,indices,cv_index,Data_Type,data_slide)

% 划分训练集,验证集,测试集
switch data_slide
    case 1 
        slice_test = (indices == cv_index);
        cv_temp = cv_index+1;
        if cv_temp>10
            cv_temp = 1;
        end
        slice_validate = (indices == cv_temp);
        slice_train = ~(xor(slice_test,slice_validate)); 
        
        Train.X = data(slice_train,:);  
        Train.Y = label(slice_train,:); 
        Train.Y_onehot = label_onehot(slice_train,:);
        
        Validate.X = data(slice_validate,:);  
        Validate.Y = label(slice_validate,:); 
        Validate.Y_onehot = label_onehot(slice_validate,:); 
        
        Test.X = data(slice_test,:);      
        Test.Y = label(slice_test,:);     
        Test.Y_onehot = label_onehot(slice_test,:);
end

%归一化处理
switch Data_Type
    case 1  % DCC
        norm_index = [1,12:23];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1),Train.X(:,2:11),TrainX_Norm_index(:,2:13)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm=[ValidateX_Norm_index(:,1),Validate.X(:,2:11),ValidateX_Norm_index(:,2:13)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1:1),Test.X(:,2:11),TestX_Norm_index(:,2:13)];    
        
    case 2  % EGSS, HTRU,PCB, ESR, ROE, OD, MCHP, EEG, RSSI
        TrainX_feature_mean = mean(Train.X,1); 
        TrainX_feature_val = var(Train.X,0,1); 
        Train.X_Norm = Normalize(Train.X,TrainX_feature_mean,TrainX_feature_val);    
        Validate.X_Norm = Normalize(Validate.X,TrainX_feature_mean,TrainX_feature_val);
        Test.X_Norm = Normalize(Test.X,TrainX_feature_mean,TrainX_feature_val);   
        
    case 3  % SE
        Train.X_Norm = Train.X/255;
        Validate.X_Norm = Validate.X/255;
        Test.X_Norm = Test.X/255;
        
    case 4 % ONP
        norm_index = [1,2,10,19:29];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1:2),Train.X(:,3:9),TrainX_Norm_index(:,3),Train.X(:,11:18),TrainX_Norm_index(:,4:14),Train.X(:,30:58)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm =  [ValidateX_Norm_index(:,1:2),Validate.X(:,3:9),ValidateX_Norm_index(:,3),Validate.X(:,11:18),ValidateX_Norm_index(:,4:14),Validate.X(:,30:58)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm =  [TestX_Norm_index(:,1:2),Test.X(:,3:9),TestX_Norm_index(:,3),Test.X(:,11:18),TestX_Norm_index(:,4:14),Test.X(:,30:58)];
        
    case 5  % OSP
        norm_index = [2,4,6:9];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [Train.X(:,1),TrainX_Norm_index(:,1),Train.X(:,3),TrainX_Norm_index(:,2),Train.X(:,5),TrainX_Norm_index(:,3:6),Train.X(:,10:17)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm = [Validate.X(:,1),ValidateX_Norm_index(:,1),Validate.X(:,3),ValidateX_Norm_index(:,2),Validate.X(:,5),ValidateX_Norm_index(:,3:6),Validate.X(:,10:17)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [Test.X(:,1),TestX_Norm_index(:,1),Test.X(:,3),TestX_Norm_index(:,2),Test.X(:,5),TestX_Norm_index(:,3:6),Test.X(:,10:17)];
        
    case 6 % BM
        norm_index = [1,4,6:10];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1),Train.X(:,2:3),TrainX_Norm_index(:,2),Train.X(:,5),TrainX_Norm_index(:,3:7),Train.X(:,11:20)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm = [ValidateX_Norm_index(:,1),Validate.X(:,2:3),ValidateX_Norm_index(:,2),Validate.X(:,5),ValidateX_Norm_index(:,3:7),Validate.X(:,11:20)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1),Test.X(:,2:3),TestX_Norm_index(:,2),Test.X(:,5),TestX_Norm_index(:,3:7),Test.X(:,11:20)];  
    
    case 7  % QSAR,IVCR, SSMCR
        Train.X_Norm = Train.X;
        Validate.X_Norm = Validate.X;
        Test.X_Norm = Test.X;
        
    case 8 % EOL
        norm_index = [2:4,7:8,11,13:14];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [Train.X(:,1),TrainX_Norm_index(:,1:3), Train.X(:,5:6),TrainX_Norm_index(:,4:5), Train.X(:,9:10),TrainX_Norm_index(:,6), Train.X(:,12),TrainX_Norm_index(:,7:8), Train.X(:,15:16)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm =  [Validate.X(:,1),ValidateX_Norm_index(:,1:3), Validate.X(:,5:6),ValidateX_Norm_index(:,4:5), Validate.X(:,9:10),ValidateX_Norm_index(:,6), Validate.X(:,12),ValidateX_Norm_index(:,7:8), Validate.X(:,15:16)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [Test.X(:,1),TestX_Norm_index(:,1:3), Test.X(:,5:6),TestX_Norm_index(:,4:5), Test.X(:,9:10),TestX_Norm_index(:,6), Test.X(:,12),TestX_Norm_index(:,7:8), Test.X(:,15:16)];   
    
    case 9 % SB
        norm_index = [1:2,4:8];
        TrainX_divi = Train.X(:,norm_index);
        ValidateX_divi = Validate.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1:2),Train.X(:,3),TrainX_Norm_index(:,3:7),Train.X(:,9)];

        ValidateX_Norm_index = Normalize(ValidateX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Validate.X_Norm = [ValidateX_Norm_index(:,1:2),Validate.X(:,3),ValidateX_Norm_index(:,3:7),Validate.X(:,9)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1:2),Test.X(:,3),TestX_Norm_index(:,3:7),Test.X(:,9)];         
end
end
end