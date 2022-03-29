function [Result_optimal_values, Result_optimal_Paras,Para_Init] = PSO_SFNN_Algorithm(Train,Validate,Para_Init,Para_Optimize)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tabulate_Y = tabulate(Train.Y) ;
Para_Init.FL_Weight = tabulate_Y(:,3)/100;  % FL聚焦损失函数的权重,即每个类别的百分比,1*N
      
%Train：同一组数据下学习48组参数,再将数据10次交叉验证
%       得到10*48 组实验结果,每行是不同数据下的48组参数的实验结果,每列是同一组参数在不同数据下的实验结果   
Result_Train = arrayfun(@(p1,p2,p3) PSO_SFNN_Algorithm_train(Train,Validate,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_train_values = [Result_Acc,Result_nodes, Traintime];  % PSO寻找的最优的Acc/隐藏层结点/Traintime
% Result_train_Para = Para_Train(TrainAcc_Index,:);

% 返回最优的 Acc对应的参数alpha,BatchSize,lambda
Result_Train_values = zeros(length(Result_Train),3); %[Acc,Hidden,Traintime]
for i = 1:length(Result_Train)
    Result_Train_tmp = Result_Train{i};
    Result_Train_values(i,:) = Result_Train_tmp.values; % 48 * 3的double
    Result_Train_Paras{i,:}   = Result_Train_tmp.Paras; % 48 * 3的double   
end
[Result_Train_values_new,Result_Train_values_index] = sortrows(Result_Train_values, [-1 -1 -1]);
Result_optimal_values = Result_Train_values_new(1,:);
Result_optimal_Paras  = Result_Train_Paras{Result_Train_values_index,:};
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Result_Train = PSO_SFNN_Algorithm_train(Train,Validate,Para_Init,alpha,BatchSize,lambda)
tic
for i = 1:Para_Init.self_max_iter 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 初始化种群
    Para_Init.self_V = unifrnd (Para_Init.self_min_v, Para_Init.self_max_v,  Para_Init.self_pN,  Para_Init.self_dim); % 随机生成粒子速度
    Para_Init.self_X = unifrnd (Para_Init.self_min_h, Para_Init.self_max_h,  Para_Init.self_pN,  Para_Init.self_dim); % 随机生成粒子位置
    Para_Init.self_pbest = Para_Init.self_X;   % 10 * 1, 每个粒子的位置 == 个体最佳位置
    
    for j  = 1:Para_Init.self_pN
        Para_Init.Hidden = ceil(Para_Init.self_pbest(j)); % 向上取整
        fprintf('Population_x=%d\n', Para_Init.Hidden)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 给定隐藏层结点数目后, 运行SFNN_K123部分【Adam优化权重和偏置；GB选择学习率、批大小、正则化系数、激活函数类型和数据分布类型】
        % 返回最优的Acc
        [Result_Train_values_init, Result_train_Paras_init] = SFNN_Algorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
        tmp_init = Result_Train_values_init(1,1);  % Acc
        Para_Init.self_pfit(j) = tmp_init;         % 当前粒子对应的适应度值 即 每个list的 err中最小的一个
        if tmp_init > Para_Init.self_gfit          % 当前粒子的全局最佳适应度值 即 err 和 当前全局最小的 err 相比，保留最小的 err 及其对应的 x
            Para_Init.self_gfit = tmp_init;        % 所有粒子的全局最佳适应度值 1 * 1, 即最小的 err
            Para_Init.self_gbest = Para_Init.self_X(j); % 所有粒子的全局最佳位置 1 * 1, 即最小的 x 的值  
            Para_Init.Para = Result_train_Paras_init;   % 当准确率高于全局gfit时,保存相应的Para
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 更新粒子位置
        % 速度更新 + 位置更新
        Para_Init.self_V(j) = Para_Init.self_V(j) + Para_Init.self_c1 * rand(1) * (Para_Init.self_pbest(j) - Para_Init.self_X(j))...
                         + Para_Init.self_c2 * rand(1) * (Para_Init.self_gbest - Para_Init.self_X(j)); % V 速度更新
        Para_Init.self_X(j) = Para_Init.self_X(j) + 0.2 * Para_Init.self_V(j);  % X 位置更新
        Para_Init.Hidden = ceil(Para_Init.self_X(j));
        fprintf('Iterator_x=%d\n', Para_Init.Hidden)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 给定隐藏层结点数目后, 运行SFNN_K123部分【Adam优化权重和偏置；GB选择学习率、批大小、正则化系数、激活函数类型和数据分布类型】
        % 返回最优的Acc
        [Result_Train_values_iter, Result_train_Paras_iter] = SFNN_Algorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
        tmp_iter = Result_Train_values_iter(1,1);  % Acc
        if tmp_iter > Para_Init.self_pfit(j)       % 更新个体最优
            Para_Init.self_pfit(j) = tmp_iter;
            Para_Init.self_pbest(j) = Para_Init.self_X(j);
            if tmp_iter > Para_Init.self_gfit  % 更新全局最优    % 当前粒子的全局最佳适应度值 即 err 和 当前全局最小的 err 相比，保留最小的 err 及其对应的 x
                Para_Init.self_gfit  = tmp_iter;                % 所有粒子的全局最佳适应度值 1 * 1, 即最小的 err
                Para_Init.self_gbest = Para_Init.self_X(j);     % 所有粒子的全局最佳位置 1 * 1, 即最小的 x 的值
                Para_Init.Para       = Result_train_Paras_iter; % 当准确率高于全局gfit时,保存相应的Para
            end
        end   
    end
end
Traintime = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_Train_values = [Train_Acc,Train_WeightF1,Train_Kappa,Train_Loss]; +++ 隐藏层结点 +++ 训练时间 
Result_Acc   = Para_Init.self_gfit;   % PSO寻找的最优的Acc
Result_nodes = Para_Init.self_gbest;  % PSO寻找的最优的隐藏层结点
Result_Para  = Para_Init.Para;        % PSO寻找的最优的参数

Result_Train.values = [Result_Acc,Result_nodes, Traintime];  % PSO寻找的最优的Acc/隐藏层结点/Traintime
Result_Train.Paras  = Result_Para;                           % PSO寻找的最优的参数
end
