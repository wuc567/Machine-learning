clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

[label,data]=libsvmread('PCB.txt');   %��ȡ�ļ�
[data_r,data_c] = size(data);
Data_Type = 2;

% ��ʼ������
s=0;   
t=1;  
para_st = [' -s ',num2str(s),' -t ',num2str(t) ];
v=10;  
init_c=1; 
init_g=0.125;

t=2.262; Data_epochs = 10;
Contrast_SVC_Result=[];SVC_Acc=[];SVC_Para=[];
indices=crossvalind('Kfold',data_r,10);

tic
for k=1:Data_epochs 
    fprintf('�����ĵ�������=%d\n',k)
    
    % ����ѵ����,���Լ�
    [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,k,Data_Type);
    
    % Ѱ�����Ų��� c �� g
    % c �ı仯��Χ�� 2^(-2),2^(-1.5),...,2^(4),
    % g �ı仯��Χ�� 2^(-4),2^(-3.5),...,2^(4)
    [bestacc,bestc,bestg] = SVMcgForClass(TrainY,TrainX_Norm,s,t,10,-1,1,1,-1,1,1,0.9);
    para_cg_best = ['-c ',num2str(bestc),' -g ',num2str(bestg) ];
    para_svc_best = [para_st, para_cg_best];
    
    % �������Ų����µ� Test ���ݼ�������ָ��
    disp('**************** Running SVC Algorithm Now ! ! ! **************************')
    tic
    para_model_best = svmtrain(TrainY,TrainX_Norm,para_svc_best);
    Traintime = toc;
    
    tic
    [predict_label, accuracy, decision_values] = svmpredict(TestY,TestX_Norm,para_model_best);
    Testtime= toc;
    
    SVC_time{k} = [Traintime,Testtime];
    SVC_Para{k} = para_svc_best;
    SVC_Predict{k} = predict_label;
    SVC_label{k} = TestY;
    SVC_Acc = [SVC_Acc,[accuracy(1);mean(predict_label == TestY)]]; % �������ȷ�ʡ��ع�ľ��������ع��ƽ�����ϵ��;
                                  % ��ʹ��Ԥѵ����ʱ��������ȡaccuracy(1,1)��Ϊ��������Ӧ��ȡ��mean(predicted_label==testlabel) 
end
[Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(SVC_time); % ����ÿ������ָ���µ�Mean,SE


disp('**************** Running Here Now. Going to end ! ! ! **************************')
SVC_Acc_Result = Search_SVC_para(t,Data_epochs,SVC_Acc);
[SVC_Acc_max,SVC_Acc_index] = max(SVC_Acc_Result(:,2));
SVC_Acc_bias = SVC_Acc_Result(SVC_Acc_index,3);

% [SVC_Acc_index,SVC_Acc_max,SVC_Acc_bias] = Search_SVC_para(t,Data_epochs,SVC_Acc);%����ÿ�е�bias,�������Acc�µ����Ų���
% % [SVC_Acc_max,SVC_Acc_index] = max(SVC_Acc);
SVC_Predict_label = SVC_Predict(:,SVC_Acc_index);
SVC_true_label = SVC_label(:,SVC_Acc_index);

if size(SVC_Predict_label,2)>2
    SVC_y_hat = SVC_Predict_label(:,1);
    SVC_y = SVC_true_label(:,1); 
else
    SVC_y_hat = SVC_Predict_label;
    SVC_y = SVC_true_label;
end

SVC_para_best = SVC_Para{SVC_Acc_index};
[Test_Weight_F1,Test_Acc,Test_Kappa] = WeightF1_Score(SVC_y{1}, SVC_y_hat{1}); 

toc
runtime = toc;
SVC_Result_Index = [Test_Acc,SVC_Acc_bias,SVC_para_best,runtime,runtime/5,Test_Weight_F1,Test_Kappa]; %���յ�ʵ����

mkdir('E:\4��Program\4��Cheng_jiayou\30��ML_Contrast\Contrast_Result');
save('E:\4��Program\4��Cheng_jiayou\30��ML_Contrast\Contrast_Result\PCB_SVC.mat',...
      'SVC_Acc_Result','SVC_Acc_max', 'SVC_Acc_bias','SVC_para_best', 'SVC_Result_Index',...
        'Contrast_Result_A_Mean_all','Contrast_Result_A_SE_all')
% save('OnlineIntention_SVC.mat', 'SVC_para_best','accuracy','bestacc','bestc','bestg')

    
% figure;
% hold on;
% plot(TestY,'o');
% plot(SVC_Predict_label,'r*');
% legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
% title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',10);

function [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,s,t,v,cmin,cmax,cstep,gmin,gmax,gstep,accstep)
%{
���룺
train_label:ѵ�����ı�ǩ����ʽҪ����svmtrain��ͬ�� 
train:ѵ��������ʽҪ����svmtrain��ͬ�� 
cmin,cmax:�ͷ�����c�ı仯��Χ������[2^cmin,2^cmax]��Χ��Ѱ����ѵĲ���c��Ĭ��ֵΪcmin=-8��cmax=8����Ĭ�ϳͷ�����c�ķ�Χ��[2^(-8),2^8]�� 
gmin,gmax:RBF�˲���g�ı仯��Χ������[2^gmin,2^gmax]��Χ��Ѱ����ѵ�RBF�˲���g��Ĭ��ֵΪgmin=-8��gmax=8����Ĭ��RBF�˲���g�ķ�Χ��[2^(-8),2^8]�� 
v:����Cross Validation�����еĲ���������ѵ��������v-fold Cross Validation��Ĭ��Ϊ3����Ĭ�Ͻ���3��CV���̡� 
cstep,gstep:���в���Ѱ����c��g�Ĳ�����С����c��ȡֵΪ2^cmin,2^(cmin+cstep),��,2^cmax,��g��ȡֵΪ2^gmin,2^(gmin+gstep),��,2^gmax��Ĭ��ȡֵΪcstep=1,gstep=1�� 
accstep:������ѡ����ͼ��׼ȷ����ɢ����ʾ�Ĳ��������С��[0,100]֮���һ��������Ĭ��Ϊ4.5�� 

�����
bestacc:����CV�����µ���ѷ���׼ȷ�ʡ� 
bestc:��ѵĲ���c�� 
bestg:��ѵĲ���g��
%}

if nargin < 10
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end

[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
eps = 1e-1;bestc = 1;bestg = 0.1;bestacc = 0;basenum = 2;

para_st = [' -s ',num2str(s), ' -t ',num2str(t) ];
for i = 1:m
    for j = 1:n
        para_cg = [' -v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        para_svc = [para_st,para_cg];
        cg(i,j) = svmtrain(train_label, train, para_svc);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
    end
end

% figure;
% [C,h] = contour(X,Y,cg,50:accstep:100);
% clabel(C,h,'Color','r');
% xlabel('log2c','FontSize',12);
% ylabel('log2g','FontSize',12);
% firstline = 'SVC����ѡ����ͼ(�ȸ���ͼ)[GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVAccuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',12);
% grid on; 
% 
% figure;
% meshc(X,Y,cg);
% % mesh(X,Y,cg);
% % surf(X,Y,cg);
% axis([cmin,cmax,gmin,gmax,30,100]);
% xlabel('log2c','FontSize',12);
% ylabel('log2g','FontSize',12);
% zlabel('Accuracy(%)','FontSize',12);
% firstline = 'SVC����ѡ����ͼ(3D��ͼ)[GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVAccuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',12);
end



function [Weight_F1,Acc,Kappa] = WeightF1_Score(Y,Y_hat)
size_Y=length(Y);
if size_Y==0
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

[ConMat,~] = confusionmat(Y,Y_hat);
sum_column = sum(ConMat,1); % �к�
sum_row = sum(ConMat,2);    % �к�

diag_ConMat = diag(ConMat);
Acc = sum(diag_ConMat)/size_Y;

pe = sum_column * sum_row/(size_Y^2);
Kappa = (Acc-pe)/(1-pe);

if any(sum_column==0) || any(sum_row==0) % �ж��Ƿ���0ֵ
    Weight_F1 = 0;
else
    P = diag_ConMat'./sum_column;
    R = diag_ConMat'./sum_row';
    F1_score = 2*P.*R./(P+R);   % ����ÿ������¶�Ӧ�� F1_Scores
    
    F1_nan = isnan(F1_score);   % �ж��Ƿ��п�ֵ
    if ismember(1,F1_nan)
        [F1_nan_row,F1_nan_column] = find(F1_nan==1);
        F1_score(F1_nan_row,F1_nan_column)=0;
    end   
    
    count=[];
    Y_unique = unique(union(unique(Y),unique(Y_hat)));
    for r = 1:length(Y_unique)
        Y_unique_num = sum(Y==Y_unique(r));
        count = [count,Y_unique_num];  %ÿ����������,1*N
    end
    Weight_F1 = sum(count.*F1_score)/size_Y;  
end
end



function [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,cv_index,Data_Type)
% ����ѵ����,��֤��,���Լ�
slice_test=(indices == cv_index);
cv_temp=cv_index+1;
if cv_temp>10
    cv_temp=randperm(10,1);
end
slice_validate = (indices == cv_temp);
slice_train = ~(xor(slice_test,slice_validate)); 

Train.X = data(slice_train,:);  
Train.Y = label(slice_train,:); 

Test.X = data(slice_test,:);       
Test.Y = label(slice_test,:);     

%��һ������
switch Data_Type
    case 1  % DCC
        norm_index = [1,12:23];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1),Train.X(:,2:11),TrainX_Norm_index(:,2:13)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1:1),Test.X(:,2:11),TestX_Norm_index(:,2:13)];    
        
    case 2  % EGSS, HTRU,PCB, ESR
        TrainX_feature_mean = mean(Train.X,1); 
        TrainX_feature_val = var(Train.X,0,1); 
        Train.X_Norm = Normalize(Train.X,TrainX_feature_mean,TrainX_feature_val);    
        Test.X_Norm = Normalize(Test.X,TrainX_feature_mean,TrainX_feature_val);   
        
    case 3  % SE
        Train.X_Norm = Train.X/255;
        Test.X_Norm = Test.X/255;
        
    case 4 % ONP
        norm_index = [1,2,10,19:29];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1:2),Train.X(:,3:9),TrainX_Norm_index(:,3),Train.X(:,11:18),TrainX_Norm_index(:,4:14),Train.X(:,30:58)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm =  [TestX_Norm_index(:,1:2),Test.X(:,3:9),TestX_Norm_index(:,3),Test.X(:,11:18),TestX_Norm_index(:,4:14),Test.X(:,30:58)];
        
    case 5  % OSP
        norm_index = [2,4,6:9];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [Train.X(:,1),TrainX_Norm_index(:,1),Train.X(:,3),TrainX_Norm_index(:,2),Train.X(:,5),TrainX_Norm_index(:,3:6),Train.X(:,10:17)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [Test.X(:,1),TestX_Norm_index(:,1),Test.X(:,3),TestX_Norm_index(:,2),Test.X(:,5),TestX_Norm_index(:,3:6),Test.X(:,10:17)];
        
    case 6 % BM
        norm_index = [1,4,6:10];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1),Train.X(:,2:3),TrainX_Norm_index(:,2),Train.X(:,5),TrainX_Norm_index(:,3:7),Train.X(:,11:20)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1),Test.X(:,2:3),TestX_Norm_index(:,2),Test.X(:,5),TestX_Norm_index(:,3:7),Test.X(:,11:20)];  
    
    case 7  %  QSAR
        Train.X_Norm = Train.X;
        Test.X_Norm = Test.X;
        
    case 8 % EOL
        norm_index = [2:4,7:8,11,13:14];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [Train.X(:,1),TrainX_Norm_index(:,1:3), Train.X(:,5:6),TrainX_Norm_index(:,4:5), Train.X(:,9:10),TrainX_Norm_index(:,6), Train.X(:,12),TrainX_Norm_index(:,7:8), Train.X(:,15:16)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [Test.X(:,1),TestX_Norm_index(:,1:3), Test.X(:,5:6),TestX_Norm_index(:,4:5), Test.X(:,9:10),TestX_Norm_index(:,6), Test.X(:,12),TestX_Norm_index(:,7:8), Test.X(:,15:16)];   
    
    case 9 % SB
        norm_index = [1:2,4:8];
        TrainX_divi = Train.X(:,norm_index);
        TestX_divi = Test.X(:,norm_index);

        TrainX_feature_mean = mean(TrainX_divi,1); 
        TrainX_feature_val = var(TrainX_divi,0,1);

        TrainX_Norm_index = Normalize(TrainX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Train.X_Norm = [TrainX_Norm_index(:,1:2),Train.X(:,3),TrainX_Norm_index(:,3:7),Train.X(:,9)];

        TestX_Norm_index = Normalize(TestX_divi,TrainX_feature_mean,TrainX_feature_val); 
        Test.X_Norm = [TestX_Norm_index(:,1:2),Test.X(:,3),TestX_Norm_index(:,3:7),Test.X(:,9)];         
end

TrainX_Norm = Train.X_Norm;
TestX_Norm = Test.X_Norm;
TrainY = Train.Y;
TestY = Test.Y;
end


function Acc_Result = Search_SVC_para(t,Data_epochs,Contrast_SVC_Result)
Acc_Result=[];
for i=1:size(Contrast_SVC_Result,1) %��
    Contrast_SVC_coulumn=[];
    for j=1:size(Contrast_SVC_Result,2) %�� 
        Contrast_KNN_per_coulumn = Contrast_SVC_Result(i,j); 
        Contrast_SVC_coulumn = [Contrast_SVC_coulumn;Contrast_KNN_per_coulumn]; %��i�������е�Ԫ��
    end   
    Acc_Mean_Matrix = mean(Contrast_SVC_coulumn(:,1));  %��1��ΪAcc
    Acc_Std_Matrix = std(Contrast_SVC_coulumn(:,1),0,1);
    Acc_bias = t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result = [Acc_Result;t,Acc_Mean_Matrix,Acc_bias];      
end
end