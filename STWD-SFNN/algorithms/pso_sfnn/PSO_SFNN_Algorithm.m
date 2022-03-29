function [Result_optimal_values, Result_optimal_Paras,Para_Init] = PSO_SFNN_Algorithm(Train,Validate,Para_Init,Para_Optimize)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tabulate_Y = tabulate(Train.Y) ;
Para_Init.FL_Weight = tabulate_Y(:,3)/100;  % FL�۽���ʧ������Ȩ��,��ÿ�����İٷֱ�,1*N
      
%Train��ͬһ��������ѧϰ48�����,�ٽ�����10�ν�����֤
%       �õ�10*48 ��ʵ����,ÿ���ǲ�ͬ�����µ�48�������ʵ����,ÿ����ͬһ������ڲ�ͬ�����µ�ʵ����   
Result_Train = arrayfun(@(p1,p2,p3) PSO_SFNN_Algorithm_train(Train,Validate,Para_Init,p1,p2,p3), Para_Optimize.alpha,Para_Optimize.BatchSize,Para_Optimize.lambda,'UniformOutput',false);  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_train_values = [Result_Acc,Result_nodes, Traintime];  % PSOѰ�ҵ����ŵ�Acc/���ز���/Traintime
% Result_train_Para = Para_Train(TrainAcc_Index,:);

% �������ŵ� Acc��Ӧ�Ĳ���alpha,BatchSize,lambda
Result_Train_values = zeros(length(Result_Train),3); %[Acc,Hidden,Traintime]
for i = 1:length(Result_Train)
    Result_Train_tmp = Result_Train{i};
    Result_Train_values(i,:) = Result_Train_tmp.values; % 48 * 3��double
    Result_Train_Paras{i,:}   = Result_Train_tmp.Paras; % 48 * 3��double   
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
    % ��ʼ����Ⱥ
    Para_Init.self_V = unifrnd (Para_Init.self_min_v, Para_Init.self_max_v,  Para_Init.self_pN,  Para_Init.self_dim); % ������������ٶ�
    Para_Init.self_X = unifrnd (Para_Init.self_min_h, Para_Init.self_max_h,  Para_Init.self_pN,  Para_Init.self_dim); % �����������λ��
    Para_Init.self_pbest = Para_Init.self_X;   % 10 * 1, ÿ�����ӵ�λ�� == �������λ��
    
    for j  = 1:Para_Init.self_pN
        Para_Init.Hidden = ceil(Para_Init.self_pbest(j)); % ����ȡ��
        fprintf('Population_x=%d\n', Para_Init.Hidden)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % �������ز�����Ŀ��, ����SFNN_K123���֡�Adam�Ż�Ȩ�غ�ƫ�ã�GBѡ��ѧϰ�ʡ�����С������ϵ������������ͺ����ݷֲ����͡�
        % �������ŵ�Acc
        [Result_Train_values_init, Result_train_Paras_init] = SFNN_Algorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
        tmp_init = Result_Train_values_init(1,1);  % Acc
        Para_Init.self_pfit(j) = tmp_init;         % ��ǰ���Ӷ�Ӧ����Ӧ��ֵ �� ÿ��list�� err����С��һ��
        if tmp_init > Para_Init.self_gfit          % ��ǰ���ӵ�ȫ�������Ӧ��ֵ �� err �� ��ǰȫ����С�� err ��ȣ�������С�� err �����Ӧ�� x
            Para_Init.self_gfit = tmp_init;        % �������ӵ�ȫ�������Ӧ��ֵ 1 * 1, ����С�� err
            Para_Init.self_gbest = Para_Init.self_X(j); % �������ӵ�ȫ�����λ�� 1 * 1, ����С�� x ��ֵ  
            Para_Init.Para = Result_train_Paras_init;   % ��׼ȷ�ʸ���ȫ��gfitʱ,������Ӧ��Para
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ��������λ��
        % �ٶȸ��� + λ�ø���
        Para_Init.self_V(j) = Para_Init.self_V(j) + Para_Init.self_c1 * rand(1) * (Para_Init.self_pbest(j) - Para_Init.self_X(j))...
                         + Para_Init.self_c2 * rand(1) * (Para_Init.self_gbest - Para_Init.self_X(j)); % V �ٶȸ���
        Para_Init.self_X(j) = Para_Init.self_X(j) + 0.2 * Para_Init.self_V(j);  % X λ�ø���
        Para_Init.Hidden = ceil(Para_Init.self_X(j));
        fprintf('Iterator_x=%d\n', Para_Init.Hidden)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % �������ز�����Ŀ��, ����SFNN_K123���֡�Adam�Ż�Ȩ�غ�ƫ�ã�GBѡ��ѧϰ�ʡ�����С������ϵ������������ͺ����ݷֲ����͡�
        % �������ŵ�Acc
        [Result_Train_values_iter, Result_train_Paras_iter] = SFNN_Algorithm(Train,Validate,Para_Init,alpha,BatchSize,lambda);
        tmp_iter = Result_Train_values_iter(1,1);  % Acc
        if tmp_iter > Para_Init.self_pfit(j)       % ���¸�������
            Para_Init.self_pfit(j) = tmp_iter;
            Para_Init.self_pbest(j) = Para_Init.self_X(j);
            if tmp_iter > Para_Init.self_gfit  % ����ȫ������    % ��ǰ���ӵ�ȫ�������Ӧ��ֵ �� err �� ��ǰȫ����С�� err ��ȣ�������С�� err �����Ӧ�� x
                Para_Init.self_gfit  = tmp_iter;                % �������ӵ�ȫ�������Ӧ��ֵ 1 * 1, ����С�� err
                Para_Init.self_gbest = Para_Init.self_X(j);     % �������ӵ�ȫ�����λ�� 1 * 1, ����С�� x ��ֵ
                Para_Init.Para       = Result_train_Paras_iter; % ��׼ȷ�ʸ���ȫ��gfitʱ,������Ӧ��Para
            end
        end   
    end
end
Traintime = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Result_Train_values = [Train_Acc,Train_WeightF1,Train_Kappa,Train_Loss]; +++ ���ز��� +++ ѵ��ʱ�� 
Result_Acc   = Para_Init.self_gfit;   % PSOѰ�ҵ����ŵ�Acc
Result_nodes = Para_Init.self_gbest;  % PSOѰ�ҵ����ŵ����ز���
Result_Para  = Para_Init.Para;        % PSOѰ�ҵ����ŵĲ���

Result_Train.values = [Result_Acc,Result_nodes, Traintime];  % PSOѰ�ҵ����ŵ�Acc/���ز���/Traintime
Result_Train.Paras  = Result_Para;                           % PSOѰ�ҵ����ŵĲ���
end
