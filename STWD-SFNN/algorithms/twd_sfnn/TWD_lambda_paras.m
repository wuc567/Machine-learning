function [TWD_Threshold_init,lambda_pp,lambda_bp,lambda_np,lambda_nn, lambda_bn, lambda_pn] = TWD_lambda_paras(TWD_cases)
while 1
    list_para = rand(1,3); 
    alpha = max(list_para);
    gamma = median(list_para);
    beta = min(list_para);
    
    if TWD_cases == 1
        lambda_pp = 0;
        lambda_nn = 0;
        lambda_pn = randperm(10,1);
        
    elseif TWD_cases == 2
        lambda_pp = 0;
        lambda_list = randperm(10,2); % 产生10个1到10之间的不重复整数,且返回前3个数
        lambda_nn = min(lambda_list);
        lambda_pn = max(lambda_list);
        
    elseif TWD_cases == 3
        lambda_list = randperm(10,2); % 产生10个1到10之间的不重复整数,且返回前3个数
        lambda_nn = 0;       
        lambda_pp = min(lambda_list);
        lambda_pn = max(lambda_list);       
    else
        lambda_list = randperm(10,3); % 产生10个1到10之间的不重复整数,且返回前3个数
        lambda_pp = min(lambda_list);
        lambda_nn = median(lambda_list);
        lambda_pn = max(lambda_list);
    end
    
    lambda_np = (1-gamma)/gamma * (lambda_pn - lambda_nn) + lambda_pp;
    lambda_bn = beta*(alpha-gamma)/(gamma*(alpha-beta)) * lambda_pn + alpha*(gamma - beta)/(gamma*(alpha-beta)) * lambda_nn;
    lambda_bp = (1-alpha)*(gamma-beta)/(gamma*(alpha-beta)) * (lambda_pn - lambda_nn) + lambda_pp;
    
    if lambda_bn > lambda_pp & lambda_bp > lambda_nn
        break;
    end
end
lambda_matrix = [lambda_pp,lambda_bp,lambda_np;lambda_nn, lambda_bn, lambda_pn];
TWD_Threshold_init = [alpha,beta,gamma];

% 验证是否满足条件
lamda_bp_pp = lambda_bp - lambda_pp;
lamda_bn_nn = lambda_bn - lambda_nn;
lamda_np_bp = lambda_np - lambda_bp;
landa_pn_bn = lambda_pn - lambda_bn;
if lamda_bp_pp * lamda_bn_nn < lamda_np_bp * landa_pn_bn
    fprintf('**************** Yes,ok! it is right ! ! ! **************************');
else
    fprintf('**************** Hello world ! ! ! **************************');
end
   

