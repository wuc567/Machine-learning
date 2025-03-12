clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load D:\Cheng_jiayou\Revise_PSO_SFNN\PSO_SFNN_v1\data_is\BM.mat
Para_Init.data_slide = 1;    % ��Ҫ���ֲ��Լ�
Para_Init.Data_Type = 6;     % ȫ����һ������
Para_Init.TWD_cases = 1;     % ���� TWD ������ͬȡֵ�µ�����

% ������Ҫ�Ż��Ĳ����б�
alpha = [0.1, 0.01];     % �ݶ��½���ѧϰ�� ,0.01,0.001
BatchSize = [128, 256]; % ����ݶ��½�������С ,128,256
lambda = [1];        % ��ʧ������������ϵ��,0.1,1,10
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % �����б�,48*3

Para_Init.self_c1 = 2;
Para_Init.self_c2 = 2;
Para_Init.self_min_h = 0;                         % ���ز�����Ŀ�����ֵ
Para_Init.self_max_h = 9;                        % ���ز�����Ŀ����Сֵ
Para_Init.self_min_v = 0;                         % �ٶȵ���Сֵ
Para_Init.self_max_v = 2;                         % �ٶȵ����ֵ
Para_Init.self_max_iter = 2;                      % ��������
Para_Init.self_pN  = 2;                           % ��������
Para_Init.self_dim = 1;                           % ����ά��
Para_Init.self_pfit  = zeros(Para_Init.self_pN);                      % ÿ���������ʷ�����Ӧֵ,����С��err
Para_Init.self_pbest = zeros(Para_Init.self_pN, Para_Init.self_dim);  % ÿ�����徭�������λ��,����С��err��Ӧ����С��x
Para_Init.self_gfit  = 0;                                   % ȫ�������Ӧֵ,����С��err
Para_Init.self_gbest = zeros(1, Para_Init.self_dim);        % ���徭����ȫ�����λ��,����С��err��Ӧ����С��x

Para_Init.s = 10;   % ���������,ǰ7����Relu����,��4����tanh����
Para_Init.t = 2;   % ���ݵķֲ�����,1�Ƿ��Ӿ��ȷֲ�,���������̬�ֲ�
Para_Init.p = 1;   % ��ͬ�ĵ��η���,1��SGD+Momentum,2��Adam,3�ǲ����������AMSgrad
Para_Init.Acc_init = 0.99; % ��ʼ����׼ȷ��
Para_Init.Loss_init = 1;  % ��ʼ������ʧֵ
Para_Init.LossFun = 2;   % ��ʧ��������,1��CE��������ʧ����,2��FL�۽���ʧ����
Para_Init.FL_Adjust = 2; % FL�۽���ʧ�����ĵ�������
Para_Init.Batch_epochs = 50; % ����С�ĵ�������
Para_Init.data_r = size(data,1); % ���ݼ���������
Para_Init.data_c = size(data,2); % ���ݼ���������
Para_Init.ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot
Para_Init.folds = 10;   % ���ݵ�10�۽�����֤

% ������֤
indices = crossvalind('Kfold',Para_Init.data_r,Para_Init.folds);
Test_Result = [];
for k = 1:Para_Init.folds 
    fprintf('���ݵĽ�����֤����=%d\n',k)
    
    % �� k ���µ����ݻ���
    [Train,Validate,Test] = Data_Partition(data,label,label_onehot,indices,k,Para_Init.Data_Type,Para_Init.data_slide);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PSO-SFNNѵ�� 2�ε���+5��������Ŀ �µ����ز�����Ŀ
    [Train_values, Train_Para,Para_Init] = PSO_SFNN_Algorithm(Train,Validate,Para_Init,Para_Optimize);
    Train_Acc    = Train_values(1,1);
    Hidden_nodes = Train_values(1,2);
    Train_time   = Train_values(1,3); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PSO-SFNN������õ����Ӳ����µ�����ѧϰ����
    tic
    [~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss] = SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                                                           Train_Para{1},Train_Para{2},Train_Para{3},Train_Para{4},...
                                                           Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    Test_time = toc;
    Test_Result_per = [Train_Acc,Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss,Hidden_nodes,Train_time, Test_time];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ����ÿ���µ�����ѧϰ����
    Test_Result = [Test_Result; Test_Result_per];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MeanSE
[Test_Result_Mean,Test_Result_SE] = Result_MeanSE(Test_Result);
mkdir('D:\Cheng_jiayou\Revise_PSO_SFNN\PSO_SFNN_v1\Contrast_Result_PSO_SFNN');
save('D:\Cheng_jiayou\Revise_PSO_SFNN\PSO_SFNN_v1\Contrast_Result_PSO_SFNN\BM_PSO_SFNN.mat',...
      'Test_Result','Test_Result_Mean', 'Test_Result_SE')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Contrast_Result_Mean,Contrast_Result_SE] = Result_MeanSE(Contrast_Result)
Contrast_Result_Mean = [];  % ���в����µ������� Mean
Contrast_Result_SE   = [];  % ���в����µ������� SE
[folds,columns] = size(Contrast_Result);
for j = 1: columns
    Contrast_Result_temp = Contrast_Result(:,j);  % �� j ��
    
    % �������в����µ������� Mean �� SE
    Contrast_Result_temp_Mean = mean(Contrast_Result_temp);                        % �� j �е������� Mean
    Contrast_Result_temp_SE   = 2 * std(Contrast_Result_temp)/sqrt(folds);       % �� j �е������� SE(95%����������)
    
    Contrast_Result_Mean      = [Contrast_Result_Mean, Contrast_Result_temp_Mean]; % ���в����µ������� Mean
    Contrast_Result_SE        = [Contrast_Result_SE,   Contrast_Result_temp_SE];   % ���в����µ������� SE
end
end


