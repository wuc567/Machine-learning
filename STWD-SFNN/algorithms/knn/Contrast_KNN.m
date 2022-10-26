clc
clear
warning off
set(0,'DefaultFigureVisible', 'off')

file_path_read = 'F:\PaperTwo\220504��papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';  % ��ȡ�ļ�·��      
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU', ...
             'DCC', 'SB', 'EOL', 'BM', 'ESR', ...
             'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % ������ļ�����
file_name_per = file_name(5);
load(['C:\Users\Lenovo\Desktop\data_is_0911\'  char(file_name_per)  '.mat'])
file_path_save = 'F:\PaperTwo\220504��papertwo-Chinese\Code-Chinese-two\UCI_file\Result_algorithms\10folds_PSO_0922\'; % ��ȡ�ļ�·��
mkdir(file_path_save) 
Data_Type = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����Ԥ��� 10cv %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folds_A = 10;
load([file_path_read  char(file_name_per)  '_TWDSFNN_'  num2str(folds_A)  'cv'  '.mat']) 
indices = indices_10cv;

firstparam = [1,2];       % ������һ��������λ���б�
secondparam = 10:10:100;  % �����ڶ���������λ���б�
[F,S] = ndgrid(firstparam,secondparam);
FF = reshape(F',size(F,1)*size(S,2),1);
SS = reshape(S',size(F,1)*size(S,2),1);
Para_list=[FF,SS]; %�����б�

t = 2.262;
Data_epochs = 10;
Contrast_KNN_Result=[];

tic
for k=1:Data_epochs 
    fprintf('�����ĵ�������=%d\n',k)
    
    % ����ѵ����,���Լ�
    [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,k,Data_Type);
    
    % [Test_Acc,Test_Weight_F1,Traintime]
    fitresult = arrayfun(@(p1,p2) fitfunction(TrainX_Norm,TrainY,TestX_Norm,TestY,p1,p2), FF,SS,'UniformOutput',false);
    Contrast_KNN_Result = [Contrast_KNN_Result,fitresult];%ÿ����ͬһ�������µĲ�ͬ����,ÿ���ǲ�ͬ�����µ�ͬһ�����
end
[Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_KNN_Result); % ����ÿ������ָ���µ�Mean,SE

[Acc_index,Acc_max,Acc_bias] = Search_KNN_para(t,Data_epochs,Contrast_KNN_Result);%����ÿ�е�bias,�������Acc�µ����Ų���
KNN_para_best = Para_list(Acc_index,:);
KNN_Result_temp = [Acc_max,Acc_bias,KNN_para_best];

toc
runtime = toc;
KNN_Result = [KNN_Result_temp,runtime,runtime/Data_epochs]; %���յ�ʵ����

disp('**************** Running Here Now ! ! ! **************************')

% ��Ѳ�����,ʵ����
Type = KNN_para_best(1); % ���빫ʽ,1=ŷʽ����,2=�н�����
K= KNN_para_best(2);     % ������Ŀ
[TrainX_Norm,TrainY,TestX_Norm,TestY] = Contrast_Data_Divide(data,label);
KNN_Result_refresh = fitfunction(TrainX_Norm,TrainY,TestX_Norm,TestY,Type,K);
save([file_path_save  char(file_name_per)  '_KNN'   '.mat'], ...
     'Contrast_KNN_Result','KNN_Result','KNN_Result_refresh', 'KNN_para_best','Contrast_Result_A_Mean_all','Contrast_Result_A_SE_all')
    


function Result_Index = fitfunction(TrainX_Norm,TrainY,TestX_Norm,TestY,Type,K)
disp('**************** Running KNN Algorithm Now ! ! ! **************************')
tic
PredictY = KNN(TestX_Norm,TrainX_Norm,TrainY,Type,K);
[Test_Weight_F1,Test_Acc,Test_Kappa] = WeightF1_Score(TestY,PredictY); 
Traintime = toc;
Result_Index = [Test_Acc,Test_Weight_F1,Traintime];
end



function  maxClass = KNN(TestX_Norm,TrainX_Norm,TrainY, Type,K)
% ******����˵��***************************
% TrainX ��С=ѵ����������*������
% TestX  ��С=���Լ�������*������
% Distance ��С=���Լ�������*ѵ����������
% K:��ҪѰ�ҵĲ��������������࣬Kȡ�������ܹ����ֲ������������Ǹ���
% *****************************************
%{
1��NN�㷨���Ǵ�ѵ�������ҵ�����������ӽ���k����¼��Ȼ��������ǵ���Ҫ���������������ݵ����
���㷨�漰3����Ҫ���أ�ѵ��������������Ƶĺ�����k�Ĵ�С�����㲽�����£�
1������룺�������Զ��󣬼�������ѵ�����е�ÿ������ľ���
2�����ھӣ�Ȧ�����������k��ѵ��������Ϊ���Զ���Ľ���
3�������ࣺ������k�����ڹ�������Ҫ������Բ��Զ������

2����������ƶȵĺ���
����Խ����ζ��������������һ������Ŀ�����Խ��,����ŷʽ���롢�н����ҵȡ�

3�������ж�
ͶƱ�������������Ӷ������������ĸ����ĵ����ͷ�Ϊ���ࡣ
��ȨͶƱ�������ݾ����Զ�����Խ��ڵ�ͶƱ���м�Ȩ������Խ����Ȩ��Խ��Ȩ��Ϊ����ƽ���ĵ�����
%}

switch Type
    case 1 %ŷʽ����
        Distance = sqrt(ones(size(TestX_Norm))*(TrainX_Norm').^2+TestX_Norm.^2*ones(size(TrainX_Norm'))-2*TestX_Norm*TrainX_Norm'); 
    case 2 %�н�����
        norm_TrainX = sqrt(sum(TrainX_Norm'.^2,1)); %��TrainXÿ��������1-����,���ش�С��1*������
        norm_TestX = sqrt(sum(TestX_Norm.^2, 2));  %��TestX ÿ��������1-����,���ش�С��������*1
        Distance = bsxfun(@rdivide,bsxfun(@rdivide,TestX_Norm*TrainX_Norm',norm_TestX),norm_TrainX); %�н����ҹ�ʽ
end

TestX_r = size(TestX_Norm,1);      %���Լ�������
sortClass = zeros(TestX_r,K); %���Լ�������*ǰk��
maxClass = zeros(TestX_r,1);  %���Լ�������*1
for i=1:TestX_r
    for j=1:K
        [~,ascend_index] = sort(Distance(i,:),'ascend'); % �Ծ�������������򣬲���ȡԭ����ֵ
        sortClass(i,j) = TrainY(ascend_index(j)); % ��ȡ���������
    end  
    maxClass(i) = mode(sortClass(i,:)); % ȡ����
end
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



function [Acc_Result_index,Acc_Result_max,Acc_Result_bias]=Search_KNN_para(t,Data_epochs,Contrast_KNN_Result)
Acc_Result=[];
for i=1:size(Contrast_KNN_Result,1) %��
    Contrast_KNN_coulumn=[];
    for j=1:size(Contrast_KNN_Result,2) %�� 
        Contrast_KNN_per_coulumn=Contrast_KNN_Result{i,j}; 
        Contrast_KNN_coulumn=[Contrast_KNN_coulumn;Contrast_KNN_per_coulumn]; %��i�������е�Ԫ��
    end   
    Acc_Mean_Matrix=mean(Contrast_KNN_coulumn(:,1));  %��1��ΪAcc
    Acc_Std_Matrix=std(Contrast_KNN_coulumn(:,1),0,1);
    Acc_bias=t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result=[Acc_Result;t,Acc_Mean_Matrix,Acc_bias];      
end
[Acc_Result_max,Acc_Result_index]=max(Acc_Result(:,2));
Acc_Result_bias=Acc_Result(Acc_Result_index,3);
end


function [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,cv_index,Data_Type)
% ����ѵ����,��֤��,���Լ�
slice_test = (indices == cv_index);
cv_temp = cv_index + 1;
if cv_temp > 10
    cv_temp = 1;
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


function [TrainX_Norm,TrainY,TestX_Norm,TestY] = Contrast_Data_Divide(data,label)
% �������ݼ�
[data_r, ~] = size(data);
data_R = randperm(data_r);  
slice = fix(data_r*0.8);  
TrainX = data(data_R(1:slice),:); 
TrainY = label(data_R(1:slice),:);  
TestX = data(data_R(slice+1:data_r),:);        
TestY = label(data_R(slice+1:data_r),:);      

% ��һ��
TrainX_Norm = TrainX;
TestX_Norm = TestX;
end
