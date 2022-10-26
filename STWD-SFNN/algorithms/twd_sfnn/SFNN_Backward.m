function [W1_Batch,b1_Batch,W2_Batch,b2_Batch,WeightF1_Batch,Acc_Batch,Kappa_Batch,Loss_Batch]=SFNN_Backward(...
                                     TrainX,TrainY,ValidateX_Norm,ValidateY_onehot,ValidateY,...
                                     Y_prob,Act_Value,Der_z,...
                                     W1,b1,W2,b2, p,alpha,iter_count,...
                                     BatchSize,s,lambda,LossFun,FL_Weight,FL_Adjust)
Train_Num=size(TrainX,2);beta1=0.9;beta2=0.999;
V_dW1= zeros(size(W1));V_db1= zeros(size(b1));V_dW2= zeros(size(W2));V_db2= zeros(size(b2));
S_dW1= zeros(size(W1));S_db1= zeros(size(b1));S_dW2= zeros(size(W2));S_db2= zeros(size(b2));
epsilon=10^(-12);Weight_F1=[];Loss=[];Para_Batch=[];Kappa=[];Acc=[];

rp = randperm(Train_Num); 
for tb=1:BatchSize:Train_Num  %iterations
    tmp=tb+BatchSize-1;
    if tmp>Train_Num  %�����һ�������������Batch_Size��Сʱ,���䵥����Ϊһ��
        Batch_TrainX=TrainX(:,rp(tb:end)); 
        Batch_TrainY=TrainY(:,rp(tb:end)); 
        Batch_Y_prob=Y_prob(:,rp(tb:end));   
        Batch_Act_Value=Act_Value(:,rp(tb:end)); 
        Batch_Der_z=Der_z(:,rp(tb:end)); 
        Batch_Num=size(Batch_TrainX,1);
    else
        Batch_TrainX=TrainX(:,rp(tb:tmp)); %����
        Batch_TrainY=TrainY(:,rp(tb:tmp)); %��ʵ��ǩ
        Batch_Y_prob=Y_prob(:,rp(tb:tmp)); %Ԥ���ǩ
        Batch_Act_Value=Act_Value(:,rp(tb:tmp));  %�����ֵ
        Batch_Der_z=Der_z(:,rp(tb:tmp)); %���������z(=W1*X+b1)��ƫ����
        Batch_Num=size(Batch_TrainX,2);
    end
    
    %�������������µ��ݶ�
    [Der_W2,Der_b2,Der_W1,Der_b1]=Parameter_Gradient(Batch_TrainX,Batch_TrainY,Batch_Y_prob,Batch_Act_Value,Batch_Der_z,...
                                                     W1,W2,lambda,LossFun,FL_Weight,FL_Adjust);
    Batch_Der_W2=1/Batch_Num.*Der_W2;
    Batch_Der_b2=1/Batch_Num.*Der_b2;
    Batch_Der_W1=1/Batch_Num.*Der_W1;
    Batch_Der_b1=1/Batch_Num.*Der_b1;
    
    if p==1
        %Adam���η�
        V_dW1 = beta1.* V_dW1 + (1-beta1).* Batch_Der_W1;
        S_dW1 = beta2.* S_dW1 + (1-beta2).* (Batch_Der_W1.^2);
        V_dW1_hat = V_dW1./(1-beta1.^iter_count);
        S_dW1_hat = S_dW1./(1-beta2.^iter_count);
        W1_Para = W1 - alpha.* V_dW1_hat./(sqrt(S_dW1_hat+epsilon));
    
        V_db1 = beta1.* V_db1 + (1-beta1).*  1/size(Batch_Der_b1,2).*sum(Batch_Der_b1,2);
        S_db1 = beta2.* S_db1 + (1-beta2).* (( 1/size(Batch_Der_b1,2).*sum(Batch_Der_b1,2)).^2);
        V_db1_hat = V_db1./(1-beta1.^iter_count);
        S_db1_hat = S_db1./(1-beta2.^iter_count);
        b1_Para = b1 - alpha.* V_db1_hat./(sqrt(S_db1_hat+epsilon));  
    
        V_dW2 = beta1.* V_dW2 + (1-beta1).* Batch_Der_W2;
        S_dW2 = beta2.* S_dW2 + (1-beta2).* (Batch_Der_W2.^2);
        V_dW2_hat = V_dW2./(1-beta1.^iter_count);
        S_dW2_hat = S_dW2./(1-beta2.^iter_count);
        W2_Para = W2 - alpha.* V_dW2_hat./(sqrt(S_dW2_hat+epsilon));
    
        V_db2 = beta1.* V_db2 + (1-beta1).* 1/size(Batch_Der_b2,2).*sum(Batch_Der_b2,2);
        S_db2 = beta2.* S_db2 + (1-beta2).* ((1/size(Batch_Der_b2,2).*sum(Batch_Der_b2,2)).^2);
        V_db2_hat = V_db2./(1-beta1.^iter_count);
        S_db2_hat = S_db2./(1-beta2.^iter_count);
        b2_Para = b2 - alpha.* V_db2_hat./(sqrt(S_db2_hat+epsilon));
    end
    
    [~,~,~,~,~,Vali_Weight_F1,Vali_Acc,Vali_Kappa,~,Vali_Loss]=SFNN_Forward(ValidateX_Norm,ValidateY_onehot,ValidateY,...
                                                   W1_Para,b1_Para,W2_Para,b2_Para,s,lambda,LossFun,FL_Weight,FL_Adjust);
%     Weight_F1=[Weight_F1;Vali_Weight_F1];
%     Acc=[Acc;Vali_Acc];
%     Kappa=[Kappa;Vali_Kappa];
%     Loss=[Loss;Vali_Loss];
%     Para_Batch=[Para_Batch;{W1,b1,W2,b2}]; %��¼ÿ��epochs��Ӧ�Ĳ���
end

Acc_Batch = Vali_Acc;
WeightF1_Batch = Vali_Weight_F1;
Kappa_Batch = Vali_Kappa;
Loss_Batch = Vali_Loss;
Para = {W1,b1,W2,b2};
W1_Batch = Para{1};
b1_Batch = Para{2};
W2_Batch = Para{3};
b2_Batch = Para{4};

% [Acc_Batch,Acc_Index]=max(Acc);
% WeightF1_Batch=Weight_F1(Acc_Index);
% Kappa_Batch=Kappa(Acc_Index);
% Loss_Batch=Loss(Acc_Index);
% Para=Para_Batch(Acc_Index,:);
% W1_Batch=Para{1};
% b1_Batch=Para{2};
% W2_Batch=Para{3};
% b2_Batch=Para{4};
clear Weight_F1 Acc  Kappa Loss Para_Batch



function [Der_L_Der_W2,Der_L_Der_b2,Der_L_Der_W1,Der_L_Der_b1]=Parameter_Gradient(X,Y,Y_prob,Activate_Value,Der_Activate_Der_z,...
                                                                                  W1,W2,lambda,LossFun,FL_Weight,FL_Adjust)
epsilon=10^(-12);
if LossFun==1  %CE��������ʧ����
    Der_L_Der_A2=Y_prob-Y;   %(�����,������)=(2,8000)   
else           %FL�۽���ʧ����
    tmp1=Y_prob-Y; %(�����,������)=(2,8000)
    tmp2=1-Y_prob; %(�����,������)=(2,8000)
    tmp3=FL_Adjust.*tmp2.^FL_Adjust; %(2,8000)
    tmp4=(tmp3).*Y_prob.*(log(Y_prob+epsilon)).*Y; %(�����,������)=(2,8000)  
    tmp5=(1-Y_prob).^FL_Adjust; %(�����,������)=(2,8000)
    tmp6=tmp1.*tmp5;  %(�����,������)=(2,8000)
    Der_L_Der_A2=FL_Weight.*(tmp4+tmp6); %(�����,������)=(2,8000)   
end

Der_L_Der_W2=Der_L_Der_A2 * Activate_Value'+lambda.*W2;%size=(�����,������)
Der_L_Der_b2=Der_L_Der_A2;  %size=(�����,������)

Der_L_Der_Activate=W2' * Der_L_Der_A2;     %size=(������,������)
Der_L_Der_A1=Der_L_Der_Activate .* Der_Activate_Der_z;%��Ԫ�����,size=(������,������)
Der_L_Der_W1=Der_L_Der_A1 * X'+lambda.*W1; %size=(������,������)
Der_L_Der_b1=Der_L_Der_A1;                 %size=(������,������)

