function [Contrast_SFNN_Result,SFNN_Result, Para_Init]= Para_Contrast_Hidden_GS(data,label,label_onehot,Para_Init,k)
Para_Init.Hidden = k;

% ������Ҫ�Ż��Ĳ����б�
alpha = [0.1,0.01,0.001];     % �ݶ��½���ѧϰ��
BatchSize = [128,256,512]; % ����ݶ��½�������С
lambda = [0,0.1,1,10];        % ��ʧ������������ϵ��
[alpha_1,BatchSize_1,lambda_1] = ndgrid(alpha,BatchSize,lambda);
Para_Optimize.alpha = reshape(alpha_1,1,[]);
Para_Optimize.BatchSize = reshape(BatchSize_1,1,[]);
Para_Optimize.lambda = reshape(lambda_1,1,[]); 
Para_Optimize.list = [Para_Optimize.alpha;Para_Optimize.BatchSize;Para_Optimize.lambda]'; % �����б�,48*3

% ��ʼ��������
Para_Init.Acc_init = 0.99; % ��ʼ����׼ȷ��
Para_Init.Loss_init = 1;  % ��ʼ������ʧֵ
Para_Init.LossFun = 2;   % ��ʧ��������,1��CE��������ʧ����,2��FL�۽���ʧ����
Para_Init.FL_Adjust = 2; % FL�۽���ʧ�����ĵ�������
Para_Init.Batch_epochs = 200; % ����С�ĵ�������
Para_Init.Data_epochs = 10;   % ���ݵĵ�������
Contrast_SFNN_Result = [];

% ������֤
indices = crossvalind('Kfold',Para_Init.data_r,10);
tic
for k = 1:Para_Init.Data_epochs 
    fprintf('���ݵĽ�����֤����=%d\n',k)
    
    %�������ݼ�
    [Train,Validate,Test] = Data_Partition(data,label,label_onehot,indices,k,Para_Init.Data_Type,Para_Init.data_slide);
         
    tabulate_Y = tabulate(Train.Y) ;
    Para_Init.FL_Weight = tabulate_Y(:,3)/100;  % FL�۽���ʧ������Ȩ��,��ÿ�����İٷֱ�,1*N
      
    %Train��ͬһ��������ѧϰ48�����,�ٽ�����10�ν�����֤
    %       �õ�10*48 ��ʵ����,ÿ���ǲ�ͬ�����µ�48�������ʵ����,ÿ����ͬһ������ڲ�ͬ�����µ�ʵ����   
    Test_Result = arrayfun(@(p1,p2,p3) ContrastAlgorithm_GS(Train,Validate,Test,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  % 1*48  
    Contrast_SFNN_Result = [Contrast_SFNN_Result;Test_Result];%ÿ����ͬһ�������µĲ�ͬ����,ÿ����ͬһ������µĲ�ͬ���ݼ�,(i,j)=[F1,Acc,Kappa],10*48    
end
[Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_SFNN_Result); % ����ÿ������ָ���µ�Mean,SE

runtime = toc;
disp('**************** Running Here Now ! ! ! **************************')
num_parameters = length(alpha) * length(BatchSize) * length(lambda);
[Para_index,Acc_bias] = Search_SFNN_para(Para_Init.Data_epochs,Contrast_SFNN_Result); % ��cell���͵�ʵ����,����ÿ�е�bias,�������Acc�µ����Ų���
SFNN_Result = [Acc_bias,Para_Optimize.list(Para_index,:),runtime,runtime/num_parameters]; %���յ�ʵ����
SFNN_Result

% ����ʵ����
mkdir('E:\4��Program\4��Cheng_jiayou\Revise_STWDSFNN\GS_SFNNN\Result_Contrast\');
save('E:\4��Program\4��Cheng_jiayou\Revise_STWDSFNN\GS_SFNN\Result_Contrast\QSAR_GSSFNN_11.mat',...
        'Contrast_SFNN_Result','SFNN_Result', 'Para_Init','Contrast_Result_A_Mean_all','Contrast_Result_A_SE_all')
end

    

function [Para_best_index,Acc_Result_bias] = Search_SFNN_para(Data_epochs,Contrast_SFNN_Result)
t=2.262;Acc_Result=[];
for i = 1:size(Contrast_SFNN_Result,2) %��
    Contrast_SFNN_row=[];
    for j = 1:size(Contrast_SFNN_Result,1) %�� 
        Contrast_SFNN_per_row = Contrast_SFNN_Result{j,i}; 
        Contrast_SFNN_row = [Contrast_SFNN_row;Contrast_SFNN_per_row]; %��i�������е�Ԫ��
    end   
    F1_Kappa_Lost{i} = Contrast_SFNN_row(:,2:4)';  % 1*48 ��cell����,ÿ������Ĵ�С�� 4*Data_epochs
    Lost_value(i,:) = Contrast_SFNN_row(:,4); % 48�� * Data_epoch��
    Acc_value(i,:) = Contrast_SFNN_row(:,1);  % 48�� * Data_epoch��
    
    Acc_Mean_Matrix = mean(Contrast_SFNN_row(:,1));  %��1��ΪAcc
    Acc_Std_Matrix = std(Contrast_SFNN_row(:,1),0,1);
    Acc_bias = t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result = [Acc_Result;t,Acc_Mean_Matrix,Acc_bias]; % �洢�ڲ�ͬ������,ͬһ�������ƽ��ʵ���� ,48*3 
end

% ����ʵ����̱���
mkdir('E:\4��Program\4��Cheng_jiayou\Revise_STWDSFNN\GS_SFNN\Result_Contrast\');
save('E:\4��Program\4��Cheng_jiayou\Revise_STWDSFNN\GS_SFNN\Result_Contrast\QSAR_GSSFNN_Para_11.mat','Acc_Result','Acc_value')


% ���ж�48�������,��ͬ�����¾�ֵ����һ����������е����� (48*3)
[Acc_Result_max_index,~] = find(Acc_Result == max(Acc_Result(:,2)));

% ����������� 2 ��,���ж�������Acc �����е�����   (48*10)
if length(Acc_Result_max_index) >= 2
    [Acc_value_max_index,~] = find(Acc_value == max(max(Acc_value))); % ���ֵ�¶�Ӧ��������,1*n
    Index_inter_Acc_Result_Acc_value = intersect(Acc_Result_max_index,Acc_value_max_index);
   
    if length(Index_inter_Acc_Result_Acc_value) >= 2
        [Lost_value_min_index,~] = find(Lost_value == min(min(Lost_value))); % ��Сֵ�¶�Ӧ��������,1*n
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
Acc_Result_bias = [Acc_value_max,Acc_Result(Para_best_index,:),F1_Kappa_Lost_value,Para_best_index,Column_best_index]; % ����Acc���ֵ,t,��ֵ,����,F1_Score, Kappa,Lost,����������,���ݵ�����������
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



