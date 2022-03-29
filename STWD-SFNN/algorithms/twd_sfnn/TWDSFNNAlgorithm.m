function Test_Result = TWDSFNNAlgorithm(Train,Validate,Test,Para_Init,alpha,BatchSize,lambda)

W1_Best=[];b1_Best=[];W2_Best=[];b2_Best=[];Para_Init.weight=[];
TWD_Next_Train = Train.X;         % 初始化边界域为整个训练集
while isempty(TWD_Next_Train)==0  % 0代表 BND 集合中仍有元素
    weight_init = size(TWD_Next_Train,1)/ Para_Init.data_r ;
    
    % 将验证集的Acc表现最好的参数传递给训练集, 并计算训练集下的网络性能
    disp('**************** From Here ! ! ! **************************')
    [Result,TWD,SFNN_wb] = SFNNAlgorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
    
    W1_Best = [W1_Best;SFNN_wb{1}]; % size(m,16)
    b1_Best = [b1_Best;SFNN_wb{2}]; % size(m,1)
    W2_Best = [W2_Best,SFNN_wb{3}]; % size(2,m)
    b2_Best = [b2_Best,SFNN_wb{4}]; % size(2,m)
    
    Para_Init.TWD_InputNum = length(TWD.TrainY); % 每次循环训练的样本量   
    if Para_Init.TWD_InputNum ==0
        Para_Init.weight = weight_init;
        break;
    end
    
    disp('**************** From Here ! ! ! **************************')
    % 将训练集下的网络隐藏层输出值和网络误差分别作为 TWD 算法的data 和 label, 并通过不断迭代确定网络的隐层结点数目 
    [TWD_Result,TWD_BND,TWD_Next_Train] = TWDAlgorithm(TWD,Para_Init);
    Train.X = Train.X(TWD_Next_Train,:);
    Train.Y = Train.Y(TWD_Next_Train,:); 
    Train.X_Norm = Train.X_Norm(TWD_Next_Train,:);
    Train.Y_onehot = Train.Y_onehot(TWD_Next_Train,:);

    if length(TWD_Next_Train) ==0
        if Para_Init.Train_num ==1
            Para_Init.weight = [Para_Init.weight;weight_init];
        else
            Para_Init.weight = [Para_Init.weight;1-size(TWD_Next_Train,1)/ Para_Init.data_r];
        end
    else
        Para_Init.weight = [Para_Init.weight;1-size(TWD_Next_Train,1)/ Para_Init.data_r];
    end   
    
    Para_Init.Train_num = Para_Init.Train_num +1;
end

disp('**************** Running Here Now ! ! ! **************************')
% 每次循环后的权重和偏置,线性组合
Para_Init.weight = diff([0;Para_Init.weight]);
W1 = Para_Init.weight .*  W1_Best;         % size(m,16)
b1 = Para_Init.weight .*  b1_Best;         % size(m,1)
W2 = Para_Init.weight'.*  W2_Best;         % size(2,m)
b2 = 1/Para_Init.Train_num .* sum(Para_Init.weight' .*  b2_Best,2); % size(2,1)
% W1 = W1_Best;         % size(m,16)
% b1 = b1_Best;         % size(m,1)
% W2 = W2_Best;
% b2 = 1/size(b2_Best,2) .* sum(b2_Best,2); % size(2,1)

% 将网络隐层结点数目加入网络拓扑结构中,并计算测试集下的网络性能
[~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss] = SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                      W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
Test_Result = [Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss,Para_Init.Train_num];
mkdir('E:\4—Program\4—Cheng_jiayou\28—Constrast_TWD\Result_Contrast\');
save('E:\4—Program\4—Cheng_jiayou\28—Constrast_TWD\Result_Contrast\SE_TWDSFNN_Para_Result_1_11.mat',...
         'Para_Init','W1','b1','W2','b2','W1_Best','b1_Best','W2_Best','b2_Best')
clear Para_Init W1 b1 W2 b2 W1_Best b1_Best W2_Best b2_Best
end



function  [SFNN_Result,TWD,SFNN_wb] = SFNNAlgorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda)
    
%初始化神经网络的权重和偏置
[W1,b1,W2,b2]=Parameter_Initialize(Para_Init.s,Para_Init.t,Para_Init.Hidden_step,Para_Init.data_c,Para_Init.ClassNum);

%StepOne：前向传播
[F_Act_Value,F_Der_z,~,F_Y_prob,~,~,~,~,~,~]=SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,...
                                W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);

%StepTwo：反向传播
TrainAcc=[];TrainLoss=[];TrainF1=[];TrainKappa=[];ValidateAcc=[];ValidateLoss=[];Para_Train=[];TrainData=[];TrainLabel=[];
iter_count=1;alpha_v1=alpha;

while iter_count< Para_Init.Batch_epochs
    
    % 训练集的Batch下选择验证集上 Acc 最高对应的参数
    [W1_Para,b1_Para,W2_Para,b2_Para,~,Bath_Acc,~,Bath_Loss]=SFNN_Backward(Train.X',Train.Y',...
           Validate.X_Norm',Validate.Y_onehot',Validate.Y,F_Y_prob,F_Act_Value,F_Der_z,W1,b1,W2,b2,Para_Init.p,alpha,iter_count,...
           BatchSize,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % 运用到整个训练集上,查看实验效果
    [Train_ActValue,~,~,~,Train_Predict,Train_Weight_F1,Train_Acc,Train_Kappa,Train_Error,Train_Loss]=SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,W1_Para,...
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
    
    TrainData=[TrainData;{Train_ActValue}];
    TrainLabel=[TrainLabel;{Train_Error}];
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

[SFNN_Result.Train_Acc,TrainAcc_Index] = max(TrainAcc);
SFNN_Result.Train_Loss = TrainLoss(TrainAcc_Index);
SFNN_Result.Train_WeightF1 = TrainF1(TrainAcc_Index);
SFNN_Result.Train_Kappa = TrainKappa(TrainAcc_Index);

TrainX_temp = TrainData(TrainAcc_Index);
TWD.TrainX_all = TrainX_temp{1};
TrainY_temp = TrainLabel(TrainAcc_Index);
TWD.TrainY_all = TrainY_temp{1};

Train.Predict = Train_Predict;
Train.Error_samples = Train.Predict~=Train.Y;  % 对比训练集的标签和预测值
TWD.TrainX = Train.X(Train.Error_samples,:);
TWD.TrainY = Train.Y(Train.Error_samples,:);
TWD.TrainY_onehot = Train.Y_onehot(Train.Error_samples,:);
TWD.TrainX_Norm = Train.X_Norm(Train.Error_samples,:);
TWD.TrainX_Disc = Train.Disc_X(Train.Error_samples,:);

SFNN_wb = Para_Train(TrainAcc_Index,:);
clear ValidateAcc ValidateLoss TrainAcc TrainLoss TrainF1 TrainKappa Para_Train TrainData TrainLabel
end



function [TWD_Result,TWD_BND,TWD_Next_Train] = TWDAlgorithm (TWD,Para_Init)

% 获取条件属性的等价类 TWD.Disc_X_Equc 
[~,~,Disc_X_index] = unique(TWD.TrainX_Disc,'rows');  
TWD.Disc_X_Equc = splitapply(@(x){x}, find(Disc_X_index), Disc_X_index); % n*1 且是按照唯一值的升序排列

% 获取决策属性的等价类 TWD.Disc_Y_Equc
[~,~,Disc_Y_index] = unique(TWD.TrainY,'rows');  
TWD.Disc_Y_Equc = splitapply(@(x){x}, find(Disc_Y_index), Disc_Y_index); % n*1

% 获取条件概率
for i = 1:length(TWD.Disc_X_Equc)             % 遍历至 X 的等价类个数
    equc_X = TWD.Disc_X_Equc{i};              % 权值X的第i个等价类
    for j = 1:length(TWD.Disc_Y_Equc)         % 遍历至 Y 的等价类个数
        equc_Y = TWD.Disc_Y_Equc{j};          % 标签Y的第j个等价类
        TWD.Pr(i,j) = length(intersect(equc_Y,equc_X))/length(equc_X); % 条件概率,size(m,n),m为data等价类个数,n为标签等价类个数
        TWD.store{i,j} = {TWD.Pr(i,j);equc_X;j}; % 存储大小=(X 等价类个数* Y等价类个数)个cell,每个cell里是3*1的cell,分别是条件概率,X等价类的样本索引,Y标签
    end
end

% 存储条件概率及其对应的X等价类的样本索引,Y标签
[Para_Init.Pr_Max,Para_Init.Pr_Index] = max(TWD.Pr,[],2);   % 找出每个等价类的最大概率值,并记录对应标签 
Para_Init.Pr_Max_Index_row_column = [(1:length(TWD.Disc_X_Equc))',Para_Init.Pr_Index]; % 行列索引值
for r = 1:length(TWD.Disc_X_Equc) 
    Index_temp = Para_Init.Pr_Max_Index_row_column(r,:);
    TWD.store_max{r} = TWD.store{Index_temp(1),Index_temp(2)}; % 取每行的最大概率值的位置索引所对应的cell{条件概率,X等价类的样本索引,Y标签}
end  % 返回 TWD.store_max 大小是 1*r

% 更新阈值对参数列表
Para_Init.TWD_Threshold_list = [];
for i=1:length(Para_Init.Pr_Max)
    twd_para = TWD_Replace_Threshold(Para_Init.Pr_Max(i), Para_Init.TWD_Threshold_init);
    Para_Init.TWD_Threshold_list=[Para_Init.TWD_Threshold_list;twd_para];  % 3*(条件概率唯一值个数)组阈值对参数
end
Para_Init.TWD_Threshold_list = unique(Para_Init.TWD_Threshold_list,'rows'); %识别不重复的阈值对参数

% 寻找最优阈值对参数
Para_Init.TWD_alpha = Para_Init.TWD_Threshold_list(:,1);
Para_Init.TWD_beta = Para_Init.TWD_Threshold_list(:,2);
Para_Init.TWD_gamma = Para_Init.TWD_Threshold_list(:,3);
Test_Result = arrayfun(@(p1,p2,p3) TWD_Result_Cost_Acc(TWD,Para_Init,p1,p2,p3), Para_Init.TWD_alpha,Para_Init.TWD_beta,Para_Init.TWD_gamma,'UniformOutput',false);  % 1*48  

Test_Result_matrix=[];
for k = 1:size(Test_Result,2) % 列
    Test_Result_per_column = Test_Result{k}; 
    Test_Result_matrix = [Test_Result_matrix,Test_Result_per_column]; % 第i列所有列的元素
end
[Test_Result_min_max,Test_Result_min_max_index] = sortrows(Test_Result_matrix,[1 -2]);        % 对两列(Cost,Acc)的矩阵,第一列升序+第二列降序排序
Para_Init.TWD_Threshold_best = Para_Init.TWD_Threshold_list(Test_Result_min_max_index,:);     %  最优阈值对参数

% 最优阈值对参数下的实验结果
TWD_Result.Cost = Test_Result_min_max(1); 
TWD_Result.Acc = Test_Result_min_max(2);

% 返回下一次循环的样本
TWD_Next_Train = Test_Result{Test_Result_min_max_index,1}{1,3};
TWD_BND = Test_Result{Test_Result_min_max_index,1}{1,4};
end



function Result_TWD = TWD_Result_Cost_Acc(TWD,Para_Init,alpha,beta,gamma)

% 自适应求解最优阈值对
Cost_POS=0; Cost_NEG=0; Cost_BND=0;TWD_POS=[];TWD_NEG=[];TWD_BND=[];
TWD_Predict = zeros(Para_Init.TWD_InputNum,1); % TWD的分类结果,可视为预测值
for i = 1:length(TWD.Disc_X_Equc)      % 遍历至data等价类的数目
    TWD.store_max_row = TWD.store_max{i};
    Pr_i = TWD.store_max_row{1};         % 条件概率
    equc_X_index = TWD.store_max_row{2}; % X 等价类的位置索引
    data_num = length(equc_X_index);     % X 等价类的样本量
    equc_Y_class = TWD.store_max_row{3}; % 对应的标签
    
    if Para_Init.TWD_InputNum >= Para_Init.TWD_ClusterNum  % 当训练的样本数>簇类数目时,直接利用(alpha,beta)划分
        if Pr_i >= alpha
            TWD_Predict(equc_X_index,:) = equc_Y_class; % 等价类的最大概率值>alpha,该等价类标签=最大概率值的位置所对应的标签
            TWD_POS = [TWD_POS;equc_X_index];           % 正域
            Cost_POS_value = sum((1-Pr_i).* data_num .* Para_Init.TWD_lambda_pn);
            Cost_POS = Cost_POS + Cost_POS_value;
            
        elseif  Pr_i <= beta
            TWD_Predict(equc_X_index,:) = equc_Y_class; % 等价类的最大概率值<beta,该等价类标签=最大概率值的位置所对应的标签
            TWD_NEG = [TWD_NEG;equc_X_index];           % 负域
            Cost_NEG_value = sum(Pr_i.* data_num .* Para_Init.TWD_lambda_np);
            Cost_NEG = Cost_NEG + Cost_NEG_value;
            
        else
            TWD_BND = [TWD_BND;equc_X_index];           % 边界域
            Cost_BND_value = sum((1-Pr_i).*data_num.* Para_Init.TWD_lambda_bn + Pr_i.*data_num.* Para_Init.TWD_lambda_bp);
            Cost_BND = Cost_BND + Cost_BND_value;   
        end
        
    else  %当训练的样本数<簇类数目时,利用gamma划分     
        if Pr_i >= gamma
            TWD_Predict(equc_X_index,:) = equc_Y_class; %等价类的最大概率值>alpha,该等价类标签=最大概率值的位置所对应的标签
            TWD_POS = [TWD_POS;equc_X_index]; %正域        
            Cost_POS_value = sum((1-Pr_i).* data_num .* Para_Init.TWD_lambda_pn);
            Cost_POS = Cost_POS + Cost_POS_value;        
            
        else
            TWD_Predict(equc_X_index,:) = equc_Y_class; %等价类的最大概率值<beta,该等价类标签=最大概率值的位置所对应的标签
            TWD_NEG = [TWD_NEG;equc_X_index]; %负域
            Cost_NEG_value = sum(Pr_i.* data_num .* Para_Init.TWD_lambda_np);
            Cost_NEG = Cost_NEG + Cost_NEG_value;                  
        end
    end
end
TWD_Error = find(TWD_Predict ~= TWD.TrainY);
if Para_Init.ClassNum == 2
    TWD_Next_Train = [TWD_Error;TWD_BND];
else
    TWD_Next_Train = [TWD_Error,TWD_NEG,TWD_BND];
end

Result_Cost = Cost_POS + Cost_NEG + Para_Init.TWD_sigma* Cost_BND;
Result_Acc = 1 - length(TWD_Error)/length(TWD.TrainY);
Result_TWD = [Result_Cost,Result_Acc,{TWD_Next_Train},{TWD_BND}];
end



function Replace_threshold = TWD_Replace_Threshold(Pr_Condition_i, Init_threshold)
 alpha = Init_threshold(1);
 beta = Init_threshold(2);
 gamma = Init_threshold(3);
 
if Pr_Condition_i == 0
    Replacement_alpha = [alpha, beta, gamma];
    Replacement_beta = [alpha, beta, gamma];
    Replacement_gamma = [alpha-beta, 0, gamma-beta];
elseif Pr_Condition_i == 1
    Replacement_alpha = [Pr_Condition_i, beta/alpha*Pr_Condition_i, gamma/alpha*Pr_Condition_i];
    Replacement_beta = [Pr_Condition_i, beta/alpha*Pr_Condition_i, gamma/alpha*Pr_Condition_i];
    Replacement_gamma = [Pr_Condition_i, beta/alpha*Pr_Condition_i, gamma/alpha*Pr_Condition_i];
else
    % 替换 alpha
    Replacement_alpha = [Pr_Condition_i, beta/alpha*Pr_Condition_i, gamma/alpha*Pr_Condition_i];
    
    % 替换 beta
    temp_alpha = alpha/beta*Pr_Condition_i;
    temp_gamma = gamma/beta*Pr_Condition_i;
    if temp_alpha>=1 | temp_gamma>=1
        Replacement_beta = [1,Pr_Condition_i,((gamma-beta)+Pr_Condition_i*(alpha-gamma))/(alpha-beta)];
    else
        Replacement_beta = [temp_alpha,Pr_Condition_i,temp_gamma];
    end
    
    % 替换 gamma
    temp_alpha1 = alpha/gamma*Pr_Condition_i;
    temp_beta = beta/gamma*Pr_Condition_i;
    if temp_alpha1>=1 | temp_beta>=1
        Replacement_gamma = [1,((beta-gamma)+Pr_Condition_i*(alpha-beta))/(alpha-gamma),Pr_Condition_i];
    else
        Replacement_gamma = [temp_alpha1,temp_beta,Pr_Condition_i];
    end       
end
Replace_threshold = [Replacement_alpha,Replacement_beta,Replacement_gamma];
if ismember(1,Replace_threshold)==1 % 阈值对中存在值为1的元素
    threshold_1_index = find(Replace_threshold==1);
    Replace_threshold(threshold_1_index) = 0.9999;
end 
if ismember(0,Replace_threshold)==1 % 阈值对中存在值为0的元素
    threshold_0_index = find(Replace_threshold==0);
    Replace_threshold(threshold_0_index) = 0.0001;
end 
end
                                  