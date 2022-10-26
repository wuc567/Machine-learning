function [Result_optimal_values, Result_optimal_Paras,Para_Init] = GS_SFNN_Algorithm(Train,Validate,Para_Init,Para_Optimize)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tabulate_Y = tabulate(Train.Y) ;
Para_Init.FL_Weight = tabulate_Y(:,3)/100;  
Result_Train = arrayfun(@(p1,p2,p3) GS_SFNN_Algorithm_train(Train,Validate,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_train_values = [Train_Acc,Train_WeightF1,Train_Kappa,Train_Loss];
% Result_train_Para = Para_Train(TrainAcc_Index,:);

% alpha,BatchSize,lambda
Result_Train_values = zeros(length(Result_Train),6); %[Acc,F1,Kappa,Loss,Hidden,Traintime]
for i = 1:length(Result_Train)
    Result_Train_tmp = Result_Train{i};
    Result_Train_values(i,:) = Result_Train_tmp.values; % 48 * 4的double
    Result_Train_Paras{i,:}   = Result_Train_tmp.Paras; % 48 * 4的double   
end
[Result_Train_values_new,Result_Train_values_index] = sortrows(Result_Train_values, [-1 -1 -1 2 2 2 ]);
Result_optimal_values = Result_Train_values_new(1,:);
Result_optimal_Paras  = Result_Train_Paras{Result_Train_values_index,:};
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Result_Train = GS_SFNN_Algorithm_train(Train,Validate,Para_Init,alpha,BatchSize,lambda)
a_0 = 0;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Para_Init.Hidden_nodes_max 
    Para_Init.Hidden = i;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Result_train_values = [Train_Acc,Train_WeightF1,Train_Kappa,Train_Loss];
    % Result_train_Para = Para_Train(TrainAcc_Index,:);
    [Result_Train_values_tmp, Result_train_Paras_tmp] = SFNN_Algorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
    a_1 = Result_Train_values_tmp(1,1);
    if a_1 - a_0 > 0
        a_0 = a_1;
    else
        Result_Train_values = Result_Train_values_tmp;
        Result_Train_Paras  = Result_train_Paras_tmp;
        break
    end 
end
Traintime = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_Train_values = [Train_Acc,Train_WeightF1,Train_Kappa,Train_Loss]; +++ 隐藏层结�? +++ 训练时间 
Result_Train.values = [Result_Train_values, Para_Init.Hidden, Traintime];
Result_Train.Paras  = Result_Train_Paras;       % GS寻找的最优的参数
end




