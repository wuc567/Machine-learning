%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TWD_lambda_threshold, BB] = STWD_lambda_matrix(num_hiddens,m,n)
alpha_old = 1;
beta_old = 0;
i = 0;
AA = [];
TWD_lambda_threshold = cell(1,num_hiddens);
while i < num_hiddens
   [~,TWD_lambda,alpha,beta] = TWD_lambda_matrix(m,n);
   a = alpha - alpha_old;
   b = beta - beta_old;
   if a <= 0  && b >= 0       % 当前阈值参数包含于上一组阈值参数
       AA = [AA; TWD_lambda(3,:)];
       [BB,BB_index] = sortrows(AA,[-1 2]);
       i = i + 1;
       alpha_old = BB(i,1);
       beta_old = BB(i,2);     
       TWD_lambda_threshold{i} = TWD_lambda;
   elseif a > 0  && b < 0      % 当前阈值参数包含 上一组阈值参数
       AA = [AA; TWD_lambda(3,:)];
       [BB,BB_index] = sortrows(AA,[-1 2]);
       CC = sortrows(BB,(2));
       if CC == BB             % 当前阈值参数完全包含 上一组阈值参数
           i = i + 1;
           alpha_old = BB(i,1);
           beta_old = BB(i,2);
           TWD_lambda_threshold{i} = TWD_lambda;
       else                     % 当前阈值参数部分包含 上一组阈值参数
           i = i + 1;
           AA(i,:) = [];
           [BB,BB_index] = sortrows(AA,[-1 2]);
           i = size(BB,1);
           alpha_old = BB(i,1);
           beta_old = BB(i,2);
       end
   end
end
TWD_lambda_threshold = TWD_lambda_threshold(:,BB_index');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [e,TWD_lambda_matrix_threshold,alpha,beta] = TWD_lambda_matrix(m,n)
% input:  lambda矩阵的大小,2行3列
% output: 三支决策满足的条件2, 生成随机的满足条件1和条件2的lambda矩阵
% steps:(1) gengrate lamda-matrix to meet inequality relation; (2)compute the threshold parameters
while 1
    TWD_lambda_matrix = rand(m,n);
    if length(unique(TWD_lambda_matrix)) ~= m * n  % 不等于6时意味着重新循环,要求是每个元素均不相等
        continue
    end
    
    lambda_np = max(TWD_lambda_matrix(1,:));
    lambda_bp = median(TWD_lambda_matrix(1,:));
    lambda_pp = 0; % min(TWD_lambda_matrix(1,:));   % 非0
    
    lambda_pn = max(TWD_lambda_matrix(2,:));
    lambda_bn = median(TWD_lambda_matrix(2,:));
    lambda_nn = 0; % min(TWD_lambda_matrix(2,:));   % 非0
    
    a = lambda_pn - lambda_bn;
    b = lambda_np - lambda_bp;
    c = lambda_bn - lambda_nn;
    d = lambda_bp - lambda_pp;
    e = a * b - c * d;
    if e >0 
        alpha = a /(a+d);
        beta  = c /(c+b); 
        gamma = (a+c)/(a+c+b+d);
        break
    else
        continue
    end 
end
TWD_lambda_matrix_threshold = [lambda_pp, lambda_bp, lambda_np;
                               lambda_pn, lambda_bn, lambda_nn;
                               alpha,     beta,      gamma];
end          

