clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load D:\Cheng_jiayou\Revise_SVC\30—ML_Contrast_v0\data_is\PCB.mat
Data_Type = 2;
[data_r, data_c]=size(data);

firstparam = 100;% 100:200:1000;  %搜索第一个参数的位置列表
secondparam = sqrt(data_c);% [data_c,log2(data_c),sqrt(data_c),0.6*data_c];  % 搜索第二个参数的位置列表
[F,S] = ndgrid(firstparam,secondparam);
FF = reshape(F',size(F,1)*size(S,2),1);
SS = reshape(S',size(F,1)*size(S,2),1);
Para_list=[FF,SS]; %参数列表

t = 2.262;
Data_epochs = 10;
Contrast_RF_Result=[];
indices=crossvalind('Kfold',data_r,10);
tic
for k=1:Data_epochs 
    fprintf('数据的交叉验证次数=%d\n',k)
    
    % 划分训练集,测试集
    [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,k,Data_Type);
    
    fitresult = arrayfun(@(p1,p2) fitfunction(TrainX_Norm,TrainY,TestX_Norm,TestY,p1,p2), FF,SS,'UniformOutput',false);
    Contrast_RF_Result = [Contrast_RF_Result,fitresult];%每列是同一组数据下的不同参数,每行是不同参数下的同一组参数
end
[Contrast_Result_A_Mean_all,Contrast_Result_A_SE_all] = Result_Mean_SE(Contrast_RF_Result); % 计算每个评价指标下的Mean,SE

[Acc_index,Acc_max,Acc_bias] = Search_RF_para(t,Data_epochs,Contrast_RF_Result);%先求每行的bias,再求最大Acc下的最优参数
RF_para_best = Para_list(Acc_index,:);
RF_Result_temp = [Acc_max,Acc_bias,RF_para_best];

toc
runtime = toc;
RF_Result = [RF_Result_temp,runtime,runtime/5]; %最终的实验结果

% 最佳参数下,实验结果
ntree = RF_para_best(1); % 弱学习器的最大迭代次数
mtry= RF_para_best(2);   % RF 划分时考虑的最大特征数
[TrainX_Norm,TrainY,TestX_Norm,TestY] = Contrast_Data_Divide(data,label);
RF_Result_refresh = fitfunction(TrainX_Norm,TrainY,TestX_Norm,TestY,ntree,mtry);
mkdir('E:\4—Program\4—Cheng_jiayou\30—ML_Contrast\Contrast_Result');
save('E:\4—Program\4—Cheng_jiayou\30—ML_Contrast\Contrast_Result\PCB_RF.mat',...
        'Contrast_RF_Result','RF_Result','RF_Result_refresh', 'RF_para_best',...
        'Contrast_Result_A_Mean_all','Contrast_Result_A_SE_all')

    
    
function [Acc_Result_index,Acc_Result_max,Acc_Result_bias] = Search_RF_para(t,Data_epochs,Contrast_RF_Result)
Acc_Result=[];
for i=1:size(Contrast_RF_Result,1) %行
    Contrast_RF_coulumn=[];
    for j=1:size(Contrast_RF_Result,2) %列 
        Contrast_RF_per_coulumn=Contrast_RF_Result{i,j}; 
        Contrast_RF_coulumn=[Contrast_RF_coulumn;Contrast_RF_per_coulumn]; %第i行所有列的元素
    end   
    Acc_Mean_Matrix=mean(Contrast_RF_coulumn(:,1));  %第1列为Acc
    Acc_Std_Matrix=std(Contrast_RF_coulumn(:,1),0,1);
    Acc_bias=t * Acc_Std_Matrix/sqrt(Data_epochs);
    Acc_Result=[Acc_Result;t,Acc_Mean_Matrix,Acc_bias];      
end
[Acc_Result_max,Acc_Result_index]=max(Acc_Result(:,2));
Acc_Result_bias=Acc_Result(Acc_Result_index,3);
end



function Result_Index = fitfunction(TrainX,TrainY,TestX,TestY,ntree,mtry)
disp('**************** Running RF Algorithm Now ! ! ! **************************')
tic
model = classRF_train(TrainX,TrainY,ntree,mtry);
Traintime = toc;

tic
[T_sim,~] = classRF_predict(TestX,model);
[Test_Weight_F1,Test_Acc,Test_Kappa] = WeightF1_Score(TestY,T_sim);
Testtime = toc;
Result_Index =[Test_Weight_F1,Test_Acc,Test_Kappa,Traintime,Testtime];
end



function [Weight_F1,Acc,Kappa] = WeightF1_Score(Y,Y_hat)
size_Y=length(Y);
if size_Y==0
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

[ConMat,~] = confusionmat(Y,Y_hat);
sum_column = sum(ConMat,1); % 列和
sum_row = sum(ConMat,2);    % 行和

diag_ConMat = diag(ConMat);
Acc = sum(diag_ConMat)/size_Y;

pe = sum_column * sum_row/(size_Y^2);
Kappa = (Acc-pe)/(1-pe);

if any(sum_column==0) || any(sum_row==0) % 判断是否有0值
    Weight_F1 = 0;
else
    P = diag_ConMat'./sum_column;
    R = diag_ConMat'./sum_row';
    F1_score = 2*P.*R./(P+R);   % 计算每个类别下对应的 F1_Scores
    
    F1_nan = isnan(F1_score);   % 判断是否有空值
    if ismember(1,F1_nan)
        [F1_nan_row,F1_nan_column] = find(F1_nan==1);
        F1_score(F1_nan_row,F1_nan_column)=0;
    end   
    
    count=[];
    Y_unique = unique(union(unique(Y),unique(Y_hat)));
    for r = 1:length(Y_unique)
        Y_unique_num = sum(Y==Y_unique(r));
        count = [count,Y_unique_num];  %每个类别的数量,1*N
    end
    Weight_F1 = sum(count.*F1_score)/size_Y;  
end
end



function [TrainX_Norm,TrainY,TestX_Norm,TestY] = Data_Partition(data,label,indices,cv_index,Data_Type)
% 划分训练集,验证集,测试集
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

%归一化处理
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
% 划分数据集
[data_r, ~] = size(data);
data_R = randperm(data_r);  
slice = fix(data_r*0.8);  
TrainX = data(data_R(1:slice),:); 
TrainY = label(data_R(1:slice),:);  
TestX = data(data_R(slice+1:data_r),:);        
TestY = label(data_R(slice+1:data_r),:);      

% 归一化
TrainX_Norm = TrainX;
TestX_Norm = TestX;
end
