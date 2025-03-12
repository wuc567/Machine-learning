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
   if a <= 0  && b >= 0       % ��ǰ��ֵ������������һ����ֵ����
       AA = [AA; TWD_lambda(3,:)];
       [BB,BB_index] = sortrows(AA,[-1 2]);
       i = i + 1;
       alpha_old = BB(i,1);
       beta_old = BB(i,2);     
       TWD_lambda_threshold{i} = TWD_lambda;
   elseif a > 0  && b < 0      % ��ǰ��ֵ�������� ��һ����ֵ����
       AA = [AA; TWD_lambda(3,:)];
       [BB,BB_index] = sortrows(AA,[-1 2]);
       CC = sortrows(BB,(2));
       if CC == BB             % ��ǰ��ֵ������ȫ���� ��һ����ֵ����
           i = i + 1;
           alpha_old = BB(i,1);
           beta_old = BB(i,2);
           TWD_lambda_threshold{i} = TWD_lambda;
       else                     % ��ǰ��ֵ�������ְ��� ��һ����ֵ����
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
% input:  lambda����Ĵ�С,2��3��
% output: ��֧�������������2, �����������������1������2��lambda����
% steps:(1) gengrate lamda-matrix to meet inequality relation; (2)compute the threshold parameters
while 1
    TWD_lambda_matrix = rand(m,n);
    if length(unique(TWD_lambda_matrix)) ~= m * n  % ������6ʱ��ζ������ѭ��,Ҫ����ÿ��Ԫ�ؾ������
        continue
    end
    
    lambda_np = max(TWD_lambda_matrix(1,:));
    lambda_bp = median(TWD_lambda_matrix(1,:));
    lambda_pp = 0; % min(TWD_lambda_matrix(1,:));   % ��0
    
    lambda_pn = max(TWD_lambda_matrix(2,:));
    lambda_bn = median(TWD_lambda_matrix(2,:));
    lambda_nn = 0; % min(TWD_lambda_matrix(2,:));   % ��0
    
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

