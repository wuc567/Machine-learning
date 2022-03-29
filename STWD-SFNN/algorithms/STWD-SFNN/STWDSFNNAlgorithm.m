function [Test_Result, SFNN_wb_all] = STWDSFNNAlgorithm(Train,Validate,Test,Para_Init,alpha,BatchSize,lambda)

tic
Cost_Result = []; Cost_Delay = []; Cost_Test = [];
W1 = []; b1 = []; W2 = []; b2 = [];
TWD_BND = Train.X;         % ��ʼ���߽���Ϊ����ѵ����
SFNN_wb_all = [];  % ���е�Ȩ�غ�ƫ��
Para_Init.Algo_run_times = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ͬһ���ݼ��£�Ȩ�غ�ƫ����ʹ����ͬ�ĳ�ʼ������,�Ա�֤���˳�����(alpha,BatchSize,lambda)���⣬����Ķ�һ�£�
% ��Ѱ�����ŵĳ��������(alpha,BatchSize,lambda)�¶�Ӧ������������ز�Ľ����Ŀ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while isempty(TWD_BND)==0  % 0���� BND ����������Ԫ��
    disp('**************** ѵ���׶� **************************')
    % disp('**************** ��ʼѰ�����ز���Ĺ��� **************************')
    
    % ����֤����Acc������õĲ������ݸ�ѵ����, ������ѵ�����µ���������
    % disp('**************** ������ **************************')
    
    [Result,STWD,SFNN_wb] = SFNNAlgorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
    SFNN_wb_all = [SFNN_wb_all; SFNN_wb];  % ���е�Ȩ�غ�ƫ��
    
    Para_Init.TWD_InputNum = length(STWD.TrainY); % ÿ��ѭ��ѵ����������   
    if Para_Init.TWD_InputNum ==0
        break;
    end
    
    % �����֧�Ĳ�����ʧ���ӳ���ʧ
    Cost_Test  = [Cost_Test, Para_Init.TWD_InputNum * sum(Para_Init.Cost_test_list(1:Para_Init.Hidden_step))];
    Cost_Delay = [Cost_Delay,Para_Init.TWD_InputNum * max(Para_Init.Cost_delay_list(1:Para_Init.Hidden_step))];
    
    % disp('**************** �����֧���� **************************')
    % ��ѵ�����µ��������ز����ֵ���������ֱ���Ϊ TWD �㷨��data �� label, ��ͨ�����ϵ���ȷ���������������Ŀ 
    % ������ֵ��
    Para_Init.STWD_lambda_threshold = Para_Init.STWD_lambda_cell{Para_Init.Hidden_step}; % ǰ������lambda����,����������ֵ��
    Para_Init.STWD_threshold = Para_Init.STWD_lambda_threshold(3,:); % ��ֵ��
    Para_Init.STWD_lambda_bp = Para_Init.STWD_lambda_threshold(1,2); % lambda_bp    
    Para_Init.STWD_lambda_np = Para_Init.STWD_lambda_threshold(1,3); % lambda_np  
    Para_Init.STWD_lambda_pn = Para_Init.STWD_lambda_threshold(2,1); % lambda_pn
    Para_Init.STWD_lambda_bn = Para_Init.STWD_lambda_threshold(2,2); % lambda_bn
    
    [TWD_Result,TWD_BND,TWD_Next_Train] = STWDAlgorithm(STWD,Para_Init);
    Train.X = Train.X(TWD_Next_Train,:);
    Train.Y = Train.Y(TWD_Next_Train,:); 
    Train.X_Norm = Train.X_Norm(TWD_Next_Train,:);
    Train.Y_onehot = Train.Y_onehot(TWD_Next_Train,:); 
    Train.Disc_X = Train.Disc_X(TWD_Next_Train,:); 
    
    % �����֧�Ľ����ʧ
    Cost_Result = [Cost_Result, cell2mat(TWD_Result.Cost)];

    if isempty(TWD_BND) == 0
        Para_Init.Hidden_step = Para_Init.Hidden_step + 1; % ���ز�����Ŀ
        Para_Init.Algo_run_times = Para_Init.Algo_run_times + 1;
    else  % �ռ�
        break
    end
    % disp('**************** ����Ѱ�����ز���Ĺ��� **************************')
end
fprintf('ѵ���׶ε����ز�����Ŀ=%d\n',Para_Init.Algo_run_times)
toc
Train_time = toc;

disp('**************** ���Խ׶� **************************')
tic
for uu = 1:Para_Init.Algo_run_times
    W1 = [W1; SFNN_wb_all{uu,1}];         % size(m,16)
    b1 = [b1; SFNN_wb_all{uu,2}];         % size(m,1)
    W2 = [W2, SFNN_wb_all{uu,3}];         % size(2,m)
    b2 = [b2, SFNN_wb_all{uu,4}];         % size(2,1)
end
b2 = 1/size(b2, 2) * sum(b2,2); % �����+��ֵ����ΪW2 * ����� + b2ʱ��ά��Ҫ��
% ��������������Ŀ�����������˽ṹ��,��������Լ��µ���������
[~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss] = SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                      W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
toc
Test_time = toc;
Test_Result = [Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss,Para_Init.Hidden_step,Train_time,Test_time,111111,Cost_Result,222222,Cost_Test,333333,Cost_Delay];

% mkdir('F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v0\Result_data\');
% save('F:\PaperTwo\Version-INs\Version-0212-0325\Experiments\Revise_STWDNN_INs\26��STWDNN_v0\Result_data\BM_STWDSFNN_Para_1_31.mat','Para_Init','W1','b1','W2','b2')
clear Para_Init W1 b1 W2 b2 W1_Best b1_Best W2_Best b2_Best
end



function  [SFNN_Result,TWD,SFNN_wb] = SFNNAlgorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda)
    
%StepTwo�����򴫲�
TrainAcc=[];TrainLoss=[];TrainF1=[];TrainKappa=[];ValidateAcc=[];ValidateLoss=[];Para_Train=[];TrainData=[];TrainLabel=[];
iter_count=1;alpha_v1=alpha;

% ��ʼ�����������Ȩ�غ�ƫ��
W1 = Para_Init.W1(Para_Init.Algo_run_times,:);   % 10*7��ֻȡ��i�������е�����
b1 = Para_Init.b1(Para_Init.Algo_run_times,:);   % 10*1
W2 = Para_Init.W2(:,Para_Init.Algo_run_times);   % 2*10
b2 = Para_Init.b2;                               % 2*1
while iter_count< Para_Init.Batch_epochs

    %StepOne��ǰ�򴫲�
    [F_Act_Value,F_Der_z,~,F_Y_prob,~,~,~,~,~,~] = SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,...
                                W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    
    % ѵ������Batch��ѡ����֤���� Acc ��߶�Ӧ�Ĳ���
    [W1_Para,b1_Para,W2_Para,b2_Para,~,Bath_Acc,~,Bath_Loss] = SFNN_Backward(Train.X',Train.Y',...
           Validate.X_Norm',Validate.Y_onehot',Validate.Y,F_Y_prob,F_Act_Value,F_Der_z,W1,b1,W2,b2,Para_Init.p,alpha,iter_count,...
           BatchSize,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % ���õ�����ѵ������,�鿴ʵ��Ч��
    [Train_ActValue,~,~,~,Train_Predict,Train_Weight_F1,Train_Acc,Train_Kappa,Train_Error,Train_Loss] = SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,W1_Para,...
           b1_Para,W2_Para,b2_Para,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % Early Stopping
    % �� Acc �� L_error ������β��仯ʱ,�ݶ��½��㷨��ѧϰ��˥����ֱ����������
    if  Para_Init.Acc_init < Train_Acc ||  Para_Init.Loss_init > Train_Loss  %������J����һ�ֵ�J_Epochs_minС,�����¿�ʼ����
        Para_Init.Loss_init = Train_Loss; %������J����ԭ���ĳ�ʼֵ
        Para_Init.Acc_init = Train_Acc;   %�µ�Acc����ԭ���ĳ�ʼֵ
        iter_count = 1;
    else
        iter_count = iter_count + 1; %��������J����һ�ֵ�J_Epochs_min����������ʱ,��ʼ�ۻ�����
        if iter_count > 10
            a = fix(iter_count/10);
            if a > 5
                break;
            end
            alpha=alpha_v1*(0.95)^(a);%��L_error��������k=20�β��½�ʱ,alphaָ��˥��
        end
    end     
     
    % Ϊ�˻���ѧϰ����
    ValidateAcc = [ValidateAcc;Bath_Acc];
    ValidateLoss = [ValidateLoss;Bath_Loss];
    
    TrainAcc = [TrainAcc;Train_Acc];  
    TrainLoss = [TrainLoss;Train_Loss]; 
    TrainF1 = [TrainF1;Train_Weight_F1];
    TrainKappa = [TrainKappa;Train_Kappa];
    
    TrainData = [TrainData;{Train_ActValue}];
    TrainLabel = [TrainLabel;{Train_Error}];
    Para_Train = [Para_Train;{W1,b1,W2,b2}]; %��¼ÿ��epochs��Ӧ�Ĳ���
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
Train.Error_samples = Train.Predict~=Train.Y;  % �Ա�ѵ�����ı�ǩ��Ԥ��ֵ
TWD.TrainX = Train.X(Train.Error_samples,:);
TWD.TrainY = Train.Y(Train.Error_samples,:);
TWD.TrainY_onehot = Train.Y_onehot(Train.Error_samples,:);
TWD.TrainX_Norm = Train.X_Norm(Train.Error_samples,:);
TWD.TrainX_Disc = Train.Disc_X(Train.Error_samples,:);

SFNN_wb = Para_Train(TrainAcc_Index,:);
clear ValidateAcc ValidateLoss TrainAcc TrainLoss TrainF1 TrainKappa Para_Train TrainData TrainLabel
end



function [TWD_Result,TWD_BND,TWD_Next_Train] = STWDAlgorithm (TWD,Para_Init)

% ��ȡ�������Եĵȼ��� TWD.TrainX_Disc 
[~,~,Disc_X_index] = unique(TWD.TrainX_Disc,'rows');  
TWD.Disc_X_Equc = splitapply(@(x){x}, find(Disc_X_index), Disc_X_index); % n*1 ���ǰ���Ψһֵ����������

% ��ȡ�������Եĵȼ��� TWD.Disc_Y_Equc
[~,~,Disc_Y_index] = unique(TWD.TrainY,'rows');  
TWD.Disc_Y_Equc = splitapply(@(x){x}, find(Disc_Y_index), Disc_Y_index); % n*1

% ��ȡ��������
for i = 1:length(TWD.Disc_X_Equc)             % ������ X �ĵȼ������
    equc_X = TWD.Disc_X_Equc{i};              % ȨֵX�ĵ�i���ȼ���
    for j = 1:length(TWD.Disc_Y_Equc)         % ������ Y �ĵȼ������
        equc_Y = TWD.Disc_Y_Equc{j};          % ��ǩY�ĵ�j���ȼ���
        TWD.Pr(i,j) = length(intersect(equc_Y,equc_X))/length(equc_X); % ��������,size(m,n),mΪdata�ȼ������,nΪ��ǩ�ȼ������
        TWD.store{i,j} = {TWD.Pr(i,j);equc_X;j}; % �洢��С=(X �ȼ������* Y�ȼ������)��cell,ÿ��cell����3*1��cell,�ֱ�����������,X�ȼ������������,Y��ǩ
    end
end

% �洢�������ʼ����Ӧ��X�ȼ������������,Y��ǩ
[Para_Init.Pr_Max,Para_Init.Pr_Index] = max(TWD.Pr,[],2);   % �ҳ�ÿ���ȼ����������ֵ,����¼��Ӧ��ǩ 
Para_Init.Pr_Max_Index_row_column = [(1:length(TWD.Disc_X_Equc))',Para_Init.Pr_Index]; % ��������ֵ
for r = 1:length(TWD.Disc_X_Equc) 
    Index_temp = Para_Init.Pr_Max_Index_row_column(r,:);
    TWD.store_max{r} = TWD.store{Index_temp(1),Index_temp(2)}; % ȡÿ�е�������ֵ��λ����������Ӧ��cell{��������,X�ȼ������������,Y��ǩ}
end  % ���� TWD.store_max ��С�� 1*r

% ������ֵ�Բ����б�
Para_Init.TWD_Threshold_list = [];
for i=1:length(Para_Init.Pr_Max)
    twd_para = TWD_Replace_Threshold(Para_Init.Pr_Max(i), Para_Init.STWD_threshold);
    Para_Init.TWD_Threshold_list = [Para_Init.TWD_Threshold_list;twd_para];  % 3*(��������Ψһֵ����)����ֵ�Բ���
end
Para_Init.TWD_Threshold_list = unique(Para_Init.TWD_Threshold_list,'rows'); %ʶ���ظ�����ֵ�Բ���

% Ѱ��������ֵ�Բ���
Para_Init.TWD_alpha = Para_Init.TWD_Threshold_list(:,1);
Para_Init.TWD_beta = Para_Init.TWD_Threshold_list(:,2);
Para_Init.TWD_gamma = Para_Init.TWD_Threshold_list(:,3);
Test_Result = arrayfun(@(p1,p2,p3) TWD_Result_Cost_Acc(TWD,Para_Init,p1,p2,p3), Para_Init.TWD_alpha,Para_Init.TWD_beta,Para_Init.TWD_gamma,'UniformOutput',false);  % 1*48  

Test_Result_matrix=[];
for k = 1:size(Test_Result,2) % ��
    Test_Result_per_column = Test_Result{k}; 
    Test_Result_matrix = [Test_Result_matrix,Test_Result_per_column]; % ��i�������е�Ԫ��
end
[Test_Result_min_max,Test_Result_min_max_index] = sortrows(Test_Result_matrix,[1 -2]);        % ������(Cost,Acc)�ľ���,��һ������+�ڶ��н�������
Para_Init.TWD_Threshold_best = Para_Init.TWD_Threshold_list(Test_Result_min_max_index,:);     %  ������ֵ�Բ���

% ������ֵ�Բ����µ�ʵ����
TWD_Result.Cost = Test_Result_min_max(1); 
TWD_Result.Acc = Test_Result_min_max(2);

% ������һ��ѭ��������
TWD_Next_Train = Test_Result{Test_Result_min_max_index,1}{1,3};
TWD_BND = Test_Result{Test_Result_min_max_index,1}{1,4};
end



function Result_TWD = TWD_Result_Cost_Acc(TWD,Para_Init,alpha,beta,gamma)

% ����Ӧ���������ֵ��
Cost_POS=0; Cost_NEG=0; Cost_BND=0;TWD_POS=[];TWD_NEG=[];TWD_BND=[];
TWD_Predict = zeros(Para_Init.TWD_InputNum,1); % TWD�ķ�����,����ΪԤ��ֵ
for i = 1:length(TWD.Disc_X_Equc)      % ������data�ȼ������Ŀ
    TWD.store_max_row = TWD.store_max{i};
    Pr_i = TWD.store_max_row{1};         % ��������
    equc_X_index = TWD.store_max_row{2}; % X �ȼ����λ������
    data_num = length(equc_X_index);     % X �ȼ����������
    equc_Y_class = TWD.store_max_row{3}; % ��Ӧ�ı�ǩ
    
    if Para_Init.TWD_InputNum >= Para_Init.TWD_ClusterNum  % ��ѵ����������>������Ŀʱ,ֱ������(alpha,beta)����
        if Pr_i >= alpha
            TWD_Predict(equc_X_index,:) = equc_Y_class; % �ȼ����������ֵ>alpha,�õȼ����ǩ=������ֵ��λ������Ӧ�ı�ǩ
            TWD_POS = [TWD_POS;equc_X_index];           % ����
            Cost_POS_value = sum((1-Pr_i).* data_num .* Para_Init.STWD_lambda_pn);
            Cost_POS = Cost_POS + Cost_POS_value;
            
        elseif  Pr_i <= beta
            TWD_Predict(equc_X_index,:) = equc_Y_class; % �ȼ����������ֵ<beta,�õȼ����ǩ=������ֵ��λ������Ӧ�ı�ǩ
            TWD_NEG = [TWD_NEG;equc_X_index];           % ����
            Cost_NEG_value = sum(Pr_i.* data_num .* Para_Init.STWD_lambda_np);
            Cost_NEG = Cost_NEG + Cost_NEG_value;
            
        else
            TWD_BND = [TWD_BND;equc_X_index];           % �߽���
            Cost_BND_value = sum((1-Pr_i).*data_num.* Para_Init.STWD_lambda_bn + Pr_i.*data_num.* Para_Init.STWD_lambda_bp);
            Cost_BND = Cost_BND + Cost_BND_value;   
        end
        
    else  %��ѵ����������<������Ŀʱ,����gamma����     
        if Pr_i >= gamma
            TWD_Predict(equc_X_index,:) = equc_Y_class; %�ȼ����������ֵ>alpha,�õȼ����ǩ=������ֵ��λ������Ӧ�ı�ǩ
            TWD_POS = [TWD_POS;equc_X_index]; %����        
            Cost_POS_value = sum((1-Pr_i).* data_num .* Para_Init.STWD_lambda_pn);
            Cost_POS = Cost_POS + Cost_POS_value;        
            
        else
            TWD_Predict(equc_X_index,:) = equc_Y_class; %�ȼ����������ֵ<beta,�õȼ����ǩ=������ֵ��λ������Ӧ�ı�ǩ
            TWD_NEG = [TWD_NEG;equc_X_index]; %����
            Cost_NEG_value = sum(Pr_i.* data_num .* Para_Init.STWD_lambda_np);
            Cost_NEG = Cost_NEG + Cost_NEG_value;                  
        end
    end
end
TWD_Error = find(TWD_Predict ~= TWD.TrainY);
if Para_Init.ClassNum == 2
    TWD_Next_Train = [TWD_Error;TWD_BND];
else
    TWD_Next_Train = [TWD_Error;TWD_NEG;TWD_BND];
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
    % �滻 alpha
    Replacement_alpha = [Pr_Condition_i, beta/alpha*Pr_Condition_i, gamma/alpha*Pr_Condition_i];
    
    % �滻 beta
    temp_alpha = alpha/beta*Pr_Condition_i;
    temp_gamma = gamma/beta*Pr_Condition_i;
    if temp_alpha>=1 | temp_gamma>=1
        Replacement_beta = [1,Pr_Condition_i,((gamma-beta)+Pr_Condition_i*(alpha-gamma))/(alpha-beta)];
    else
        Replacement_beta = [temp_alpha,Pr_Condition_i,temp_gamma];
    end
    
    % �滻 gamma
    temp_alpha1 = alpha/gamma*Pr_Condition_i;
    temp_beta = beta/gamma*Pr_Condition_i;
    if temp_alpha1>=1 | temp_beta>=1
        Replacement_gamma = [1,((beta-gamma)+Pr_Condition_i*(alpha-beta))/(alpha-gamma),Pr_Condition_i];
    else
        Replacement_gamma = [temp_alpha1,temp_beta,Pr_Condition_i];
    end       
end
Replace_threshold = [Replacement_alpha,Replacement_beta,Replacement_gamma];
if ismember(1,Replace_threshold)==1 % ��ֵ���д���ֵΪ1��Ԫ��
    threshold_1_index = find(Replace_threshold==1);
    Replace_threshold(threshold_1_index) = 0.9999;
end 
if ismember(0,Replace_threshold)==1 % ��ֵ���д���ֵΪ0��Ԫ��
    threshold_0_index = find(Replace_threshold==0);
    Replace_threshold(threshold_0_index) = 0.0001;
end 
end
                                  