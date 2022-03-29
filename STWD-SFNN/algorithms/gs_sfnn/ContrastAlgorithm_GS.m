function Test_Result = ContrastAlgorithm_GS(Train,Validate,Test,Para_Init,alpha,BatchSize,lambda)

tic
%��ʼ���������Ȩ�غ�ƫ��
[W1,b1,W2,b2]=Parameter_Initialize(Para_Init.s,Para_Init.t,Para_Init.Hidden,...
                                   Para_Init.data_c,Para_Init.ClassNum);

%StepOne��ǰ�򴫲�
[F_Act_Value,F_Der_z,~,F_Y_prob,~,~,~,~,~,~] = SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,...
                                W1,b1,W2,b2,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);

%StepTwo�����򴫲�
TrainF1=[];TrainAcc=[];TrainKappa=[];TrainLoss=[];ValidateAcc=[];ValidateLoss=[];Para_Train=[];
iter_count=1;alpha_v1 = alpha;
while iter_count< Para_Init.Batch_epochs
    
    % ѵ������Batch��ѡ����֤���� Acc ��߶�Ӧ�Ĳ���
    [W1_Para,b1_Para,W2_Para,b2_Para,~,Bath_Acc,~,Bath_Loss]=SFNN_Backward(Train.X',Train.Y',...
           Validate.X_Norm',Validate.Y_onehot',Validate.Y,F_Y_prob,F_Act_Value,F_Der_z,W1,b1,W2,b2,Para_Init.p,alpha,iter_count,...
           BatchSize,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % ���õ�����ѵ������,�鿴ʵ��Ч��
    [~,~,~,~,~,Train_Weight_F1,Train_Acc,Train_Kappa,~,Train_Loss]=SFNN_Forward(Train.X_Norm',Train.Y_onehot',Train.Y,W1_Para,...
           b1_Para,W2_Para,b2_Para,Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);
    
    % Early Stopping
    % �� Acc �� L_error ������β��仯ʱ,�ݶ��½��㷨��ѧϰ��˥����ֱ����������
    if  Para_Init.Acc_init < Train_Acc ||  Para_Init.Loss_init > Train_Loss  %������J����һ�ֵ�J_Epochs_minС,�����¿�ʼ����
        Para_Init.Loss_init = Train_Loss; %������J����ԭ���ĳ�ʼֵ
        Para_Init.Acc_init = Train_Acc;   %�µ�Acc����ԭ���ĳ�ʼֵ
        iter_count=1;
    else
        iter_count=iter_count+1; %��������J����һ�ֵ�J_Epochs_min����������ʱ,��ʼ�ۻ�����
        if iter_count>10
            a=fix(iter_count/10);
            if a>5
                break;
            end
            alpha=alpha_v1*(0.95)^(a);%��L_error��������k=20�β��½�ʱ,alphaָ��˥��
        end
    end     
     
    % Ϊ�˻���ѧϰ����
    ValidateAcc=[ValidateAcc;Bath_Acc];
    ValidateLoss=[ValidateLoss;Bath_Loss];
    
    TrainAcc=[TrainAcc;Train_Acc];  
    TrainLoss=[TrainLoss;Train_Loss]; 
    TrainF1=[TrainF1;Train_Weight_F1];
    TrainKappa=[TrainKappa;Train_Kappa];
    
    Para_Train=[Para_Train;{W1,b1,W2,b2}]; %��¼ÿ��epochs��Ӧ�Ĳ���
end
Traintime = toc;

% figure
% x=1:size(TrainAcc,1);
% plot(x,ValidateAcc,'k:',x,TrainAcc,'r-.')
% legend('Validata Acc','Train Acc')
% 
% hold on
% plot(x,ValidateLoss,'b--',x,TrainLoss,'g')
% legend('Validata Loss','Train Loss')

tic;
[Train_Acc,TrainAcc_Index]=max(TrainAcc);
Train_Loss=TrainLoss(TrainAcc_Index);
Train_WeightF1=TrainF1(TrainAcc_Index);
Train_Kappa=TrainKappa(TrainAcc_Index);
Train_Result = [Train_WeightF1,Train_Acc,Train_Kappa,Train_Loss];

Para = Para_Train(TrainAcc_Index,:);
clear ValidateF1 ValidateAcc ValidateKappa ValidateLoss TrainF1 TrainAcc TrainKappa TrainLoss Para_Train

[~,~,~,~,~,Test_WeightF1,Test_Acc,Test_Kappa,~,Test_Loss]=SFNN_Forward(Test.X_Norm',Test.Y_onehot',Test.Y,...
                      Para{1},Para{2},Para{3},Para{4},Para_Init.s,lambda,Para_Init.LossFun,Para_Init.FL_Weight,Para_Init.FL_Adjust);

Testtime = toc;
Test_Result = [Test_Acc,Test_WeightF1,Test_Kappa,Test_Loss,Para_Init.Hidden,Traintime,Testtime];

% Reault_SFNN_index = [Train_Result;Test_Result];
% xlswrite('E:\4��Program\2��MatalabCode\V11��TWDSFNN',Reault_SFNN_index)

end
                                        
             