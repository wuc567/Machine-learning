function [Contrast_SFNN_Result,SFNN_Result, Para_Init]= Para_Contrast_Hidden_K123(data,label,label_onehot,Para_Init,k,iter)
Para_Init.Hidden = k;

% 构建需要优化的参数列表
alpha = [0.1,0.01,0.001];     % 梯度下降的学习率
BatchSize = [128,256,512]; % 随机梯度下降的批大小
lambda = [0,0.1,1,10];        % 损失函数的正则项系数
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % 参数列表,48*3

% 初始化超参数
Para_Init.data_slide = 1;    % 需要划分测试集
Para_Init.Data_Type  = 7;     % 全部归一化处理
Para_Init.s = 1;   % 激活函数类型,前7个是Relu函数,后4个是tanh函数
Para_Init.t = 1;   % 数据的分布类型,1是服从均匀分布,否则服从正态分布
Para_Init.p = 1;   % 不同的调参方法,1是SGD+Momentum,2是Adam,3是不带修正项的AMSgrad
Para_Init.Acc_init = 0.99; % 初始化的准确率
Para_Init.Loss_init = 1;  % 初始化的损失值
Para_Init.LossFun = 2;   % 损失函数类型,1是CE交叉熵损失函数,2是FL聚焦损失函数
Para_Init.FL_Adjust = 2; % FL聚焦损失函数的调整因子
Para_Init.Batch_epochs = 200; % 批大小的迭代次数
Para_Init.Data_epochs = 10;   % 数据的迭代次数
Contrast_SFNN_Result = [];

% 交叉验证
indices = crossvalind('Kfold',Para_Init.data_r,10);
tic
for k = 1:Para_Init.Data_epochs 
    fprintf('数据的交叉验证次数=%d\n',k)
    
    %划分数据集
    [Train,Validate,Test] = Data_Partition(data,label,label_onehot,indices,k,Para_Init.Data_Type,Para_Init.data_slide);
         
    tabulate_Y = tabulate(Train.Y) ;
    Para_Init.FL_Weight = tabulate_Y(:,3)/100;  % FL聚焦损失函数的权重,即每个类别的百分比,1*N
      
    %Train：同一组数据下学习48组参数,再将数据10次交叉验证
    %       得到10*48 组实验结果,每行是不同数据下的48组参数的实验结果,每列是同一组参数在不同数据下的实验结果   
    Test_Result = arrayfun(@(p1,p2,p3) ContrastAlgorithm_K123(Train,Validate,Test,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  % 1*48  
    Contrast_SFNN_Result = [Contrast_SFNN_Result;Test_Result];%每行是同一组数据下的不同参数,每列是同一组参数下的不同数据集,(i,j)=[F1,Acc,Kappa],10*48    
end
runtime = toc;
disp('**************** Running Here Now ! ! ! **************************')
num_parameters = length(alpha) * length(BatchSize) * length(lambda);
[Para_index,Acc_bias] = Search_SFNN_para(Para_Init.Data_epochs,Contrast_SFNN_Result,iter); % 对cell类型的实验结果,先求每列的bias,再求最大Acc下的最优参数
SFNN_Result = [Acc_bias,Para_Optimize.list(Para_index,:),runtime,runtime/num_parameters]; %最终的实验结果
SFNN_Result

% 保存实验结果
if iter==1
    Para_Init.Hidden_K1 = Para_Init.Hidden;
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_K1_11.mat',...
        'Contrast_SFNN_Result','SFNN_Result', 'Para_Init')
elseif iter==2
    Para_Init.Hidden_K2 = Para_Init.Hidden;
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_K2_11.mat',...
        'Contrast_SFNN_Result','SFNN_Result', 'Para_Init')    
else
    Para_Init.Hidden_K3 = Para_Init.Hidden;
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_K3_11.mat',...
        'Contrast_SFNN_Result','SFNN_Result', 'Para_Init')
end
end

    

function [Para_best_index,Acc_Result_bias] = Search_SFNN_para(Data_epochs,Contrast_SFNN_Result,iter)
t=2.262;Acc_Result=[];
for i = 1:size(Contrast_SFNN_Result,2) %列
    Contrast_SFNN_row=[];
    for j = 1:size(Contrast_SFNN_Result,1) %行 
        Contrast_SFNN_per_row = Contrast_SFNN_Result{j,i}; 
        Contrast_SFNN_row = [Contrast_SFNN_row;Contrast_SFNN_per_row]; %第i列所有列的元素
    end   
    F1_Kappa_Lost{i} = Contrast_SFNN_row(:,2:4)';  % 1*48 的cell数组,每个数组的大小是 4*Data_epochs
    Lost_value(i,:) = Contrast_SFNN_row(:,4); % 48行 * Data_epoch列
    Acc_value(i,:) = Contrast_SFNN_row(:,1);  % 48行 * Data_epoch列
    
    Acc_Mean_Matrix = mean(Contrast_SFNN_row(:,1));  %第1列为Acc
    Acc_Std_Matrix = std(Contrast_SFNN_row(:,1),0,1);
    Acc_bias = t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result = [Acc_Result;t,Acc_Mean_Matrix,Acc_bias]; % 存储在不同数据下,同一组参数的平均实验结果 ,48*3 
end

% 保存实验过程变量
if iter==1
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_Para_K1_11.mat','Acc_Result','Acc_value')
elseif iter==2
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_Para_K2_11.mat','Acc_Result','Acc_value')
else
    mkdir('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\');
    save('E:\4—Program\4—Cheng_jiayou\27—K123\Result_Contrast\QSAR_SFNN_Para_K3_11.mat','Acc_Result','Acc_value')
end

% 先判断48组参数中,不同数据下均值最大的一组参数所在行的索引 (48*3)
[Acc_Result_max_index,~] = find(Acc_Result == max(Acc_Result(:,2)));

% 如果索引超过 2 个,再判断在最大的Acc 所在行的索引   (48*10)
if length(Acc_Result_max_index) >= 2
    [Acc_value_max_index,~] = find(Acc_value == max(max(Acc_value))); % 最大值下对应的行索引,1*n
    Index_inter_Acc_Result_Acc_value = intersect(Acc_Result_max_index,Acc_value_max_index);
   
    if length(Index_inter_Acc_Result_Acc_value) >= 2
        [Lost_value_min_index,~] = find(Lost_value == min(min(Lost_value))); % 最小值下对应的行索引,1*n
        Index_inter_Acc_Result_Acc_value_Lost_value = intersect(Index_inter_Acc_Result_Acc_value,Lost_value_min_index);
        
        if length(Index_inter_Acc_Result_Acc_value_Lost_value) >=1
            Para_best_index = Index_inter_Acc_Result_Acc_value_Lost_value(1);
        else
            Para_best_index = Index_inter_Acc_Result_Acc_value(1);
        end
        
    elseif length(Index_inter_Acc_Result_Acc_value) ==1
        Para_best_index = Index_inter_Acc_Result_Acc_value;
    else
        Para_best_index = Acc_Result_max_index(1);
    end
else
    Para_best_index = Acc_Result_max_index;
end

[Acc_value_max,Acc_value_max_Para_best_index] = max(Acc_value(Para_best_index,:));  
if length(Acc_value_max_Para_best_index) >=2
    [~, Lost_value_min_Para_best_index] = min(Lost_value(Para_best_index,:));
    Index_inter_Acc_value_max_Lost_value_min_index = intersect(Acc_value_max_Para_best_index,Lost_value_min_Para_best_index);
    if length(Index_inter_Acc_value_max_Lost_value_min_index) >=1
        Column_best_index = Index_inter_Acc_value_max_Lost_value_min_index(1);
    else
        Column_best_index = Acc_value_max_Para_best_index(1);
    end
else 
    Column_best_index = Acc_value_max_Para_best_index;
end
    
F1_Kappa_Lost_temp = F1_Kappa_Lost{Para_best_index};
F1_Kappa_Lost_value = F1_Kappa_Lost_temp(:,Column_best_index)';
Acc_Result_bias = [Acc_value_max,Acc_Result(Para_best_index,:),F1_Kappa_Lost_value,Para_best_index,Column_best_index]; % 包含Acc最大值,t,均值,方差,F1_Score, Kappa,Lost,参数组索引,数据迭代次数索引
clear Acc_value Acc_Result Lost_value K_value F1_Kappa_Lost_K
end



function [Train,Validate,Test] = Data_Partition(data,label,label_onehot,indices,cv_index,Data_Type,data_slide)

% 划分训练集,验证集,测试集
switch data_slide
    case 1 
        slice_test = (indices == cv_index);
        cv_temp = cv_index+1;
        if cv_temp>10
            cv_temp = randperm(10,1);
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
        
    case 2
        slice_validate = (indices == cv_index);
        slice_train = ~(slice_validate); 
        
        Train.X = data(slice_train,:);  
        Train.Y = label(slice_train,:); 
        Train.Y_onehot = label_onehot(slice_train,:);
        
        Validate.X = data(slice_validate,:);  
        Validate.Y = label(slice_validate,:); 
        Validate.Y_onehot = label_onehot(slice_validate,:); 
        
        Test.X = data(6599:7074,:);      
        Test.Y = label(6599:7074,:);     
        Test.Y_onehot = label_onehot(6599:7074,:);   
    
    case 3
        slice_validate=(indices == cv_index);
        slice_train = ~(slice_validate); 
        
        Train.X = data(slice_train,:);  
        Train.Y = label(slice_train,:); 
        Train.Y_onehot = label_onehot(slice_train,:);
        
        Validate.X = data(slice_validate,:);  
        Validate.Y = label(slice_validate,:); 
        Validate.Y_onehot = label_onehot(slice_validate,:); 
        
        Test.X = data(4340:4839,:);      
        Test.Y = label(4340:4839,:);     
        Test.Y_onehot = label_onehot(4340:4839,:);   
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
        
    case 2  % EGSS, HTRU,PCB, ESR
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
    
    case 7  %  QSAR
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



function Test_Result =ContrastAlgorithm_K123(Train,Validate,Test,Para_Init,alpha,BatchSize,lambda)
                                   
%初始化神经网络的权重和偏置
[W1,b1,W2,b2]=Parameter_Initialize(Para_Init.s,Para_Init.t,Para_Init.Hidden,...
                                   Para_Init.data_c,Para_Init.ClassNum);

%StepOne：前向传播
[F_Act_Value,F_Der_z,~,F_Y_prob,~,~,~,~,~,~]=SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,...
                                W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);

%StepTwo：反向传播
TrainF1=[];TrainAcc=[];TrainKappa=[];TrainLoss=[];ValidateAcc=[];ValidateLoss=[];Para_Train=[];
iter_count=1;alpha_v1=alpha;
while iter_count< Para_Init.Batch_epochs
    
    % 训练集的Batch下选择验证集上 Acc 最高对应的参数
    [W1_Para,b1_Para,W2_Para,b2_Para,~,Bath_Acc,~,Bath_Loss]=SFNN_Backward(Train.X',Train.Y',...
           Validate.X_Norm',Validate.Y_onehot',Validate.Y,F_Y_prob,F_Act_Value,F_Der_z,W1,b1,W2,b2,Para_Init.p,alpha,iter_count,...
           BatchSize,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % 运用到整个训练集上,查看实验效果
    [~,~,~,~,~,Train_Weight_F1,Train_Acc,Train_Kappa,~,Train_Loss]=SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,W1_Para,...
           b1_Para,W2_Para,b2_Para,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % Early Stopping
    % 当 Acc 或 L_error 连续多次不变化时,梯度下降算法的学习率衰减，直至满足条件
    if  Para_Init.Acc_init < Train_Acc ||  Para_Init.Loss_init > Train_Loss  %新来的J比上一轮的J_Epochs_min小,则重新开始计数
        Para_Init.Loss_init = Train_Loss; %新来的J覆盖原来的初始值
        Para_Init.Acc_init = Train_Acc;   %新的Acc覆盖原来的初始值
        iter_count=1;
    else
        iter_count=iter_count+1; %当新来的J比上一轮的J_Epochs_min大或两者相等时,则开始累积计数
        if iter_count>10
            a=fix(iter_count/10);
            if a>5
                break;
            end
            alpha=alpha_v1*(0.95)^(a);%当L_error满足连续k=20次不下降时,alpha指数衰减
        end
    end     
     
    % 为了绘制学习曲线
    ValidateAcc=[ValidateAcc;Bath_Acc];
    ValidateLoss=[ValidateLoss;Bath_Loss];
    
    TrainAcc=[TrainAcc;Train_Acc];  
    TrainLoss=[TrainLoss;Train_Loss]; 
    TrainF1=[TrainF1;Train_Weight_F1];
    TrainKappa=[TrainKappa;Train_Kappa];
    
    Para_Train=[Para_Train;{W1,b1,W2,b2}]; %记录每次epochs对应的参数
end
% figure
% x=1:size(TrainAcc,1);
% plot(x,ValidateAcc,'k:',x,TrainAcc,'r-.')
% legend('Validata Acc','Train Acc')
% 
% hold on
% plot(x,ValidateLoss,'b--',x,TrainLoss,'g')
% legend('Validata Loss','Train Loss')

[Train_Acc,TrainAcc_Index]=max(TrainAcc);
Train_Loss=TrainLoss(TrainAcc_Index);
Train_WeightF1=TrainF1(TrainAcc_Index);
Train_Kappa=TrainKappa(TrainAcc_Index);
Train_Result = [Train_WeightF1,Train_Acc,Train_Kappa,Train_Loss];

Para=Para_Train(TrainAcc_Index,:);
clear ValidateF1 ValidateAcc ValidateKappa ValidateLoss TrainF1 TrainAcc TrainKappa TrainLoss Para_Train

[~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss]=SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                      Para{1},Para{2},Para{3},Para{4},Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
Test_Result = [Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss];

% Reault_SFNN_index = [Train_Result;Test_Result];
% xlswrite('E:\4—Program\2—MatalabCode\V11—TWDSFNN',Reault_SFNN_index)

end
                                        
             