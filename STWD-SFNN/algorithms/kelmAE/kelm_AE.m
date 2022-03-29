%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Result = kelm_AE(train_data,train_target,test_data,test_target,Para_Init,alpha,kernel_type)
tic
s  = Para_Init.smooth_para;  % 平滑参数
C1 = Para_Init.regular_C1;   % First  ELM module的正则化因子
C2 = Para_Init.regular_C2;   % Second ELM module的正则化因子
kernel_para = Para_Init.kernel_paras; % 核函数的各种参数

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBF_kernel:  k(x,y) = exp(-1/C.*||x-y||^2);
% lin_kernel:  k(x,y) = x'* y + C;
% poly_kernel: k(x,y) = (a * x'* y + C)^d;         % a=1,d=2
% wav_kernel:  k(x,y) = cos(A/B.* x) * exp(-x./C)  % A/B=1.75
% kernel_para = [C,B,A]; % [20,2,3.5]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the non-equilibrium label completion matrix construction
Conf = NeLC(train_target,alpha,s);

% First ELM module
Ytrain = (train_target' * Conf);
[num_class,~] = size(test_target);
Xtrain = [train_data,sum(Ytrain,2)/num_class];
[X,TX] = felm_kernel(test_data,Xtrain,train_data,C1, kernel_type,kernel_para);
clear num_class  Xtrain  train_data  test_data

% Second ELM module
[~,TY] = selm_kernel(TX,X,train_target,C2, kernel_type,kernel_para,Conf);
clear Conf Ytrain  train_target

% prediction
Outputs = TY';
Y_pred = logical(Outputs==max(Outputs));  % argmax后的网络输出,size=(类别数,样本量)
Y_hat = vec2ind(Y_pred)';                 % 将热编码转换成标签,Y_hat的size=(样本量,1)
Y = vec2ind(test_target)';
[Weight_F1,Acc,Kappa] = WeightF1_Score(Y,Y_hat);
Train_time = toc;
Result = [Acc,Weight_F1,Kappa,Train_time];

% Pre_Labels = sign(Outputs);
% ret.HL = Hamming_loss(Pre_Labels,test_target);
% ret.RL = Ranking_loss(Outputs,test_target);
% ret.OE = One_error(Outputs,test_target);
% ret.CV = coverage(Outputs,test_target);
% ret.AP = Average_precision(Outputs,test_target);

clear Outputs TY  test_target  Pre_Labels 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Weight_F1,Acc,Kappa] = WeightF1_Score(Y,Y_hat)
size_Y = length(Y);
if size_Y == 0
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