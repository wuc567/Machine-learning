clear
clc
rng(0)
warning off
set(0,'DefaultFigureVisible', 'off')

load C:\Users\Lenovo\Desktop\data_is\ONP.mat

Para_Init.data_slide = 1;    % ��Ҫ���ֲ��Լ�
Para_Init.Data_Type = 4;     % ȫ����һ������
Para_Init.TWD_cases = 1;     % ���� TWD ������ͬȡֵ�µ�����
Para_Init.data_r = size(data,1); % ���ݼ���������
Para_Init.data_c = size(data,2); % ���ݼ���������
Para_Init.ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

% ʶ�����ݼ����������ͣ�Int�ͻ���Double��
[~, Para_Init.feature_double] = data_type_recognize(data);

% ������Ҫ�Ż��Ĳ����б�
alpha = [0.1,0.01,0.001];     % �ݶ��½���ѧϰ��
BatchSize = [128,256];        % ����ݶ��½�������С ,512
lambda = [0,0.1,1];        % ��ʧ������������ϵ�� ,10
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % �����б�,48*3

% ��ʼ��������
Para_Init.s = 1;  % ���������,ǰ7����Relu����,��4����tanh����
Para_Init.t = 1;   % ���ݵķֲ�����,1�Ƿ��Ӿ��ȷֲ�,���������̬�ֲ�
Para_Init.p = 1;   % ��ͬ�ĵ��η���,1��Adam,2��SGD+Momentum,3�ǲ����������AMSgrad
Para_Init.Acc_init = 0.9; % ��ʼ����׼ȷ��
Para_Init.Loss_init = 1;  % ��ʼ������ʧֵ
Para_Init.LossFun = 2;    % ��ʧ��������,1��CE��������ʧ����,2��FL�۽���ʧ����
Para_Init.FL_Adjust = 2;      % FL�۽���ʧ�����ĵ�������
Para_Init.Batch_epochs = 20;  % ����С�ĵ�������
Para_Init.Data_epochs = 10;   % ���ݵĵ�������
Para_Init.init_threshold_epochs  = 5;  % ��ʼ�������Ĵ���
Para_Init.TWD_ClusterNum = 5; % ��ɢ�����̵Ĵ�����Ŀ
Para_Init.TWD_sigma = 2;  
Para_Init.Hidden_step = 1;     % ��ʼ�����ز���Ϊ1
Para_Init.Hidden_up = 10;      % ���ز�����Ŀ���Ͻ�
Para_Init.Algo_run_times = 1;  % �㷨�����д���
clear TWD_threshold_pair TWD_alpha_init  TWD_beta_init TWD_gamma_init alpha BatchSize lambda alpha_1 BatchSize_1 lambda_1

% ������֤
Para_Init.indices = crossvalind('Kfold',Para_Init.data_r,10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����ÿ�ε����ݼ���ѵ��������֤���Ͳ��Լ�����һ�����������ݼ���ʵ������Ӱ�죬����Ϊ��10�۽�����֤
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ��ʼ��Ȩ�� w(10*n, 2*10) ��ƫ�� b(10*1,2*1) 
for iii = 1:Para_Init.init_threshold_epochs  % ��ʼ�������Ĵ���
    [Weight_1,bias_1,Weight_2,bias_2] = Parameter_Initialize(Para_Init.s,Para_Init.t,Para_Init.Hidden_up,Para_Init.data_c,Para_Init.ClassNum);    
    Para_Init.Weight_1{iii} = Weight_1;
    Para_Init.bias_1{iii} = bias_1;
    Para_Init.Weight_2{iii} = Weight_2;
    Para_Init.bias_2{iii} = bias_2;
end

Contrast_Result_10cv = [];
for k = 1:Para_Init.Data_epochs 
    Contrast_SFNN_Result=[]; Contrast_SFNN_Para = [];
    fprintf('���ݵĽ�����֤����=%d\n',k)
    
    % ��ʼ�������ֵ��
    m = 2; n = 3;
    [Para_Init.STWD_lambda_cell, Para_Init.STWD_threshold] = STWD_lambda_matrix(Para_Init.Hidden_up,m,n);
    
    % �������ݼ�
    [Train,Validate,Test] = Data_Partition(data,label,label_onehot,Para_Init.indices,k,Para_Init.Data_Type,Para_Init.data_slide);
    
    % ��ɢ�����ݼ�
    Train.Disc_X = Data_discrete(Train.X_Norm,Para_Init);
    
    % ������������֧���ӳ���ʧֵ�Ͳ�����ʧֵ
    Para_Init.Cost_test_list   = linspace(1,50,Para_Init.Hidden_up); % [1,50]��Χ�ڵ� 1 * 10 �����������Ĳ�����ʧ�����
    Para_Init.Cost_delay_list  = linspace(1,50,Para_Init.Hidden_up); % [1,50]��Χ�ڵ� 1 * 10 �������������ӳ���ʧ�����
    
    tabulate_Y = tabulate(Train.Y) ;
    Para_Init.FL_Weight = tabulate_Y(:,3)/100;  % FL�۽���ʧ������Ȩ��,��ÿ�����İٷֱ�,1*N
    
%     Test_Result = STWDSFNNAlgorithm(Train,Validate,Test,Para_Init,0.1,128,0);
%     Contrast_SFNN_Result = [Contrast_SFNN_Result;Test_Result];  
    %Train��ͬһ��������ѧϰ48�����,�ٽ�����10�ν�����֤
    %       �õ�10*48 ��ʵ����,ÿ���ǲ�ͬ�����µ�48�������ʵ����,ÿ����ͬһ������ڲ�ͬ�����µ�ʵ����   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ��ͬһ���ݼ��£���ÿһ���£�Ϊ��ɸѡ�����ŵĳ��������(alpha,BatchSize,lambda)����֤ÿ��ʹ�õ�����ĳ�������ͬ
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for kk = 1:Para_Init.init_threshold_epochs  % ��ʼ�������Ĵ���
        fprintf('��ʼ�������Ĵ���=%d\n',kk)
        Para_Init.W1 = Para_Init.Weight_1{kk};
        Para_Init.b1 = Para_Init.bias_1{kk};
        Para_Init.W2 = Para_Init.Weight_2{kk};
        Para_Init.b2 = Para_Init.bias_2{kk};
        
        % Test_Result = [Acc,F1,Kappa,Loss,Hidden_step,Train,Test,111111,Cost_Result,222222,Cost_Test,333333,Cost_Delay]
        [Test_Result, SFNN_wb_all] = arrayfun(@(p1,p2,p3) STWDSFNNAlgorithm(Train,Validate,Test,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  % 1*48  
        Contrast_SFNN_Result = [Contrast_SFNN_Result; Test_Result]; % ÿ����ͬһ�������µĲ�ͬ����,ÿ����ͬһ������µĲ�ͬ���ݼ�,(i,j)=[F1,Acc,Kappa],kk*48   
        Contrast_SFNN_Para = [Contrast_SFNN_Para; SFNN_wb_all]; % ÿ����ͬһ�������µĲ�ͬ����,ÿ����ͬһ������µĲ�ͬ���ݼ�,(i,j)=[F1,Acc,Kappa],kk*48 
        disp('=================================================')    
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Ѱ�� ��ʼ�������Ĵ��� * (alpha * Batchsize * lambda) = 5 * 36��cell�����ŵ�һ�鳬����
    % disp('**************** Running Here Now. Going to end ! ! ! **************************')
    
    % Contrast_SFNN_Result = [Acc,F1,Kappa,Loss,Hidden_step,Train,Test,111111,Cost_Result,222222,Cost_Test,333333,Cost_Delay]
    % Acc_bias = [acc,[t,mean,se], F1_Kappa_Lost_K_Train_Test_value, ����������,���ݵ�����������]
    [Para_index,Acc_bias] = Search_TWDSFNN_para(Para_Init.init_threshold_epochs,Contrast_SFNN_Result); % ��cell���͵�ʵ����,����ÿ�е�bias,�������Acc�µ����Ų���
    STWDNN_Cost_vector = Contrast_SFNN_Result{Acc_bias(11),Acc_bias(12)}(8:end);      % 111111,Cost_Result,222222,Cost_Test,333333,Cost_Delay
    STWDNN_Cost_matrix = reshape(STWDNN_Cost_vector,length(STWDNN_Cost_vector)/3,3)'; % [[111111,Cost_Result];[222222,Cost_Test];[333333,Cost_Delay]]
    STWDNN_Cost_matrix = STWDNN_Cost_matrix(:,2:end); % ֻȡcostֵ,[Cost_Result; Cost_Test; Cost_Delay]
    
    % Acc_bias = [acc,[t,mean,se], F1_Kappa_Lost_K_Train_Test_value, ����������,���ݵ�����������]
    % ����Acc_bias,STWDNN_Cost_vector,STWDNN_Cost_matrix,[alpha,BatchSize,lambda],W1,b1,W2,b2
    STWDNN_Result_per = {Acc_bias, STWDNN_Cost_vector, STWDNN_Cost_matrix, Para_Optimize.list(Para_index,:), Contrast_SFNN_Para{Para_index}}; %���յ�ʵ����
    Contrast_Result_10cv = [Contrast_Result_10cv; STWDNN_Result_per];
    disp('=================================================')
end
[Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all, Contrast_Result_K_Mean, Contrast_Result_K_SE,Contrast_Result_optimal] = Result_Mean_SE(Contrast_Result_10cv); % ����ÿ������ָ���µ�Mean,SE

% ����ʵ����
mkdir('F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_data\');
save( 'F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_data\ONP_STWDSFNN_11.mat',...
     'Contrast_Result_10cv', 'Para_Init','Contrast_Result_A_Mean_all','Contrast_Result_A_SE_all',...
     'Contrast_Result_K_Mean','Contrast_Result_K_SE','Contrast_Result_optimal')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���� exptime ��ʵ�������� Acc �¶�Ӧ�� cost ����ͼ
figure;
STWDNN_Cost_mat = Contrast_Result_optimal{3};
x = 1:size(STWDNN_Cost_mat,2);
Cost_R_opt     = STWDNN_Cost_mat(1,:); % y1-- Result cost
Cost_test_opt  = STWDNN_Cost_mat(2,:); % y2-- Test cost
Cost_delay_opt = STWDNN_Cost_mat(3,:); % y3-- Delay cost

p1 = plot(x, Cost_R_opt,    'b*-', 'LineWidth', 1.5); % 'linestyle','*-', 'Color','b'
hold on
p2 = plot(x, Cost_test_opt, 'k^-', 'LineWidth', 1.5); % 'linestyle','^-', 'Color','k',
hold on
p3 = plot(x, Cost_delay_opt,'ro-', 'LineWidth', 1.5); % 'linestyle','o-', 'Color','g',

xlabel('Number of attributes')
ylabel('Cost')
title('Trends of costs under different number of attributes')
h = legend([p1,p2,p3],'Result cost ','Test cost','Delay cost','Location','NorthWest'); % ʾ���������Ͻ�
set(h,  'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
set(gca,'FontName','Times New Roman','FontSize',17,'FontWeight','normal')
mkdir(      'F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_Figure\')
saveas(gcf,['F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_Figure\ONP_Cost_trend_11','.jpg']);

 

% Acc_Result_bias = [acc,[t,mean,se], F1_Kappa_Lost_K_Train_Test_value, ����������,���ݵ�����������]
function [Para_best_index,Acc_Result_bias] = Search_TWDSFNN_para(Data_epochs,Contrast_SFNN_Result)
t=2.262;Acc_Result=[]; 
for i = 1:size(Contrast_SFNN_Result,1)   % ��, i = 1,2,...10
    Contrast_SFNN_row=[];
    for j=1:size(Contrast_SFNN_Result,2) % ��,j = 1,2,...,48
        %%%%%%%%%%%%%%%%%%%%%%%% [Acc, F1, Kappa, Loss, K, Train_time, Test_time] %%%%%%%%%%%%%%%%%%%%%%%%
        Contrast_SFNN_per_row = Contrast_SFNN_Result{i,j}(1:7); 
        Contrast_SFNN_row = [Contrast_SFNN_row;Contrast_SFNN_per_row]; %��i�������е�Ԫ��
    end   
    
    %%%%%%%%%%%%%%%%%%%%%%%% [Acc, F1, Kappa, Loss, K, Train_time, Test_time] %%%%%%%%%%%%%%%%%%%%%%%%
    Lost_value(i,:) = Contrast_SFNN_row(:,4);      % 48�� * Data_epoch��
    K_value(i,:) = Contrast_SFNN_row(:,5);         % 48�� * Data_epoch��
    F1_Kappa_Lost_K_Train_Test{i} = Contrast_SFNN_row(:,2:7)';  % 1*48 ��cell����,ÿ������Ĵ�С�� 4*Data_epochs
    
    Acc_value(i,:) = Contrast_SFNN_row(:,1);       % 48�� * Data_epoch��
    Acc_Mean_Matrix= mean(Contrast_SFNN_row(:,1)); % ��1��ΪAcc
    Acc_Std_Matrix = std(Contrast_SFNN_row(:,1),0,1);
    Acc_bias = t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result = [Acc_Result;t,Acc_Mean_Matrix,Acc_bias]; % �洢�ڲ�ͬ������,ͬһ�������ƽ��ʵ���� ,48*3     
end

% ����ʵ����̱���
mkdir('F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_data\');
save('F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v1\Result_data\ONP_STWDSFNN_Para_11.mat','Acc_Result')

% ���ж�48�������,��ͬ�����¾�ֵ����һ����������е����� (48*3)
[Acc_Result_max_index,~] = find(Acc_Result == max(Acc_Result(:,2))); % ������

% ����������� 2 ��,���ж������� Acc �����е�����   (48*10)
if length(Acc_Result_max_index) >= 2
    [Acc_value_max_index,~] = find(Acc_value == max(max(Acc_value))); % ���ֵ�¶�Ӧ��������,1*n
    Index_inter_Acc_Result_Acc_value = intersect(Acc_Result_max_index,Acc_value_max_index);
    
% ����������� 2 ��,���ж�����С�� K �����е�����   (48*10)
    if length(Index_inter_Acc_Result_Acc_value) >= 2
        [K_value_min_index,~] = find(K_value == min(min(K_value))); % ��Сֵ�¶�Ӧ��������,1*n
        Index_inter_Acc_Result_Acc_value_K_value = intersect(Index_inter_Acc_Result_Acc_value,K_value_min_index);

% ����������� 2 ��,���ж�����С�� Lost �����е�����   (48*10)
        if length(Index_inter_Acc_Result_Acc_value_K_value) >=2
            [Lost_value_min_index,~] = find(Lost_value == min(min(Lost_value))); % ��Сֵ�¶�Ӧ��������,1*n
            Index_inter_Acc_Result_Acc_value_K_value_Lost_value = intersect(Index_inter_Acc_Result_Acc_value_K_value,Lost_value_min_index);

% ����������� 2 ��,����һ��Ľ��������ѡȡһ��Ԫ��,��¼�����е�����   (48*10)            
            if length(Index_inter_Acc_Result_Acc_value_K_value_Lost_value) >=1
                 Para_best_index = Index_inter_Acc_Result_Acc_value_K_value_Lost_value(1);
            else
               Para_best_index =  Index_inter_Acc_Result_Acc_value_K_value(1);
            end
            
        elseif length(Index_inter_Acc_Result_Acc_value_K_value) ==1
            Para_best_index = Index_inter_Acc_Result_Acc_value_K_value;
        else
            Para_best_index = Index_inter_Acc_Result_Acc_value(1);
        end
        
    elseif length(Index_inter_Acc_Result_Acc_value) == 1
        Para_best_index = Index_inter_Acc_Result_Acc_value;
    else
        Para_best_index = Acc_Result_max_index(1);
    end
    
else
    Para_best_index = Acc_Result_max_index; 
end

[Acc_value_max,Acc_value_max_Para_best_index] = max(Acc_value(Para_best_index,:));

% ����������� 2 ��,���ж�����С�� K �����е�����   (48*10)
if length(Acc_value_max_Para_best_index) >=2
    [~, K_value_min_Para_best_index] = min(K_value(Para_best_index,:));
    Index_inter_Acc_value_max_K_value_min_index = intersect(Acc_value_max_Para_best_index,K_value_min_Para_best_index);

% ����������� 2 ��,���ж�����С�� Lost �����е�����   (48*10)
    if length(Index_inter_Acc_value_max_K_value_min_index) >=2
        [~, Lost_value_min_Para_best_index] = min(Lost_value(Para_best_index,:));
        Index_inter_Acc_value_max_K_value_min_Lost_value_min_index = intersect(Index_inter_Acc_value_max_K_value_min_index,Lost_value_min_Para_best_index);

% ����������� 2 ��,����һ��Ľ��������ѡȡһ��Ԫ��,��¼�����е�����   (48*10)         
        if length(Index_inter_Acc_value_max_K_value_min_Lost_value_min_index) >=1
            Column_best_index = Index_inter_Acc_value_max_K_value_min_Lost_value_min_index(1);
        else
            Column_best_index = Index_inter_Acc_value_max_K_value_min_index(1);
        end
            
    elseif length(Index_inter_Acc_value_max_K_value_min_index)==1
        Column_best_index = Index_inter_Acc_value_max_K_value_min_index;
    else
        Column_best_index = Acc_value_max_Para_best_index(1);
    end
        
else
    Column_best_index = Acc_value_max_Para_best_index;
end

F1_Kappa_Lost_K_Train_Test_temp = F1_Kappa_Lost_K_Train_Test{Para_best_index};
F1_Kappa_Lost_K_Train_Test_value = F1_Kappa_Lost_K_Train_Test_temp(:,Column_best_index)';
Acc_Result_bias = [Acc_value_max,Acc_Result(Para_best_index,:),F1_Kappa_Lost_K_Train_Test_value,Para_best_index,Column_best_index]; % [acc,[t,mean,se], F1_Kappa_Lost_K_Train_Test_value, ����������,���ݵ�����������]
clear Acc_value Acc_Result Lost_value K_value F1_Kappa_Lost_K
end
 


function [Train,Validate,Test] = Data_Partition(data,label,label_onehot,indices,cv_index,Data_Type,data_slide)

% ����ѵ����,��֤��,���Լ�
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

%��һ������
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [feature_list_int,feature_list_double] = data_type_recognize(data)
feature_list_int    = [];
feature_list_double = [];
for i = 1:size(data,2)
    data_col = data(:,i);
    data_col_x = floor(data_col);
    if data_col == data_col_x    % ����
        feature_list_int = [feature_list_int,i];
    else
        feature_list_double = [feature_list_double,i];
    end 
end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Disc_X = Data_discrete(data,Para_Init)

% Disc_X�� ��ɢ���������
attribute_list = 1:Para_Init.data_c;

% ������ɢ��
Disc_X = [];
for i = 1:length(attribute_list)
    attribute_list_temp = attribute_list(i);                       % �� i ������
    if ismember(attribute_list_temp,Para_Init.feature_double) % �� i ��������ʵֵ��
        [Disc_X_temp,~] = kmeans(data(:,attribute_list_temp), Para_Init.TWD_ClusterNum,'Distance','sqeuclidean','Replicates',5);
        Disc_X = [Disc_X, Disc_X_temp];
    else
        Disc_X = [Disc_X, data(:,attribute_list_temp)]; % �����б����ɢ�����
    end       
end
end