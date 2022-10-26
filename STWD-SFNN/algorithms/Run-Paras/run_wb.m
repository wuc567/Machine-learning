clc
clear
rng(0) 

file_path_read = 'C:\Users\lenovo\Desktop\data_is_0911\';
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU', ...
                    'DCC', 'SB', 'EOL', 'BM', 'ESR', ...
                    'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % ������ļ�����
file_name_per = file_name(5); 
load([file_path_read  char(file_name_per)   '.mat'])
file_path_save = 'F:\PaperTwo\220504��papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';
mkdir(file_path_save)

%%%%%%%%%%%%%%%%%%%%% ���ݼ���10�ۻ��� %%%%%%%%%%%%%%%%%%%%%
data_r = size(data,1);    % ���ݼ���������
indices_5cv = crossvalind('Kfold',  data_r, 10);
str_list_one = strcat(file_path_save,  char(file_name_per),  '_TWDSFNN_10cv',  '.mat');
save(str_list_one,  'indices_10cv');

%%%%%%%%%%%%%%%%%%%%% ������Ȩ�غ�ƫ�� %%%%%%%%%%%%%%%%%%%%%
s = 1;
t = 1;
K = 50;
InputDimension = size(data,2);    % ��������������
ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
[Weight_1,bias_1,Weight_2,bias_2]=Parameter_Initialize(s,t,K,InputDimension,ClassNum);
str_list_two = strcat(file_path_save,  char(file_name_per),  '_SFNN_para_',  '11',  '.mat');
save(str_list_two, 'Weight_1','bias_1', 'Weight_2', 'bias_2');

%%%%%%%%%%%%%%%%%%%%% ��֧���ߵ���ֵ�� %%%%%%%%%%%%%%%%%%%%%
TWD_cases = 1;
[threshold_init,lambda_matrix] = TWD_lambda_paras_v2(TWD_cases);
str_list_three = strcat(file_path_save, char(file_name_per),'_TWDSFNN_para_twd', '.mat');
save(str_list_three, 'threshold_init', 'lambda_matrix');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Weight_1,bias_1,Weight_2,bias_2]=Parameter_Initialize(s,t,K,InputDimension,ClassNum)
%K=1; %K�����ӵ�Ȩ������,�����ӵ�������Ŀ
if s<8 
    if t==1 
        Weight_1=(rand(K,InputDimension)*2-1)*sqrt(6/InputDimension);%Relu����,���Ӿ��ȷֲ�,����㵽����
        Weight_2=(rand(ClassNum,K)*2-1)*sqrt(6/K);%Relu����,���Ӿ��ȷֲ�,���㵽�����
    else
        Weight_1=(normrnd(0,sqrt(2/InputDimension),[K,InputDimension]));%Relu����,������̬�ֲ�,����㵽����
        Weight_2=(normrnd(0,sqrt(2/K),[ClassNum,K]));%Relu����,������̬�ֲ�,���㵽�����
    end
else
    if t==1
        Weight_1=(rand(K,InputDimension)*2-1)*sqrt(6/(InputDimension+K));%tanh����,���Ӿ��ȷֲ�,����㵽����
        Weight_2=(rand(ClassNum,K)*2-1)*sqrt(6/(K+ClassNum));%tanh����,���Ӿ��ȷֲ�,���㵽�����
    else
        Weight_1=(normrnd(0,sqrt(2/(InputDimension+K)),[K,InputDimension]));%tanh����,������̬�ֲ�,����㵽����
        Weight_2=(normrnd(0,sqrt(2/(K+ClassNum)),[ClassNum,K]));%tanh����,������̬�ֲ�,���㵽�����
    end
end
bias_1 = rand(K,1)*0.01;
bias_2 = rand(ClassNum,1)*0.01;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TWD_Threshold_init,TWD_lambda_matrix] = TWD_lambda_paras_v2(TWD_cases)
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
        lambda_list = randperm(10,2); % ����10��1��10֮��Ĳ��ظ�����,�ҷ���ǰ3����
        lambda_nn = min(lambda_list);
        lambda_pn = max(lambda_list);
        
    elseif TWD_cases == 3
        lambda_nn = 0;
        lambda_list = randperm(10,2); % ����10��1��10֮��Ĳ��ظ�����,�ҷ���ǰ3����
        lambda_pp = min(lambda_list);
        lambda_pn = max(lambda_list);       
    else
        lambda_list = randperm(10,3); % ����10��1��10֮��Ĳ��ظ�����,�ҷ���ǰ3����
        lambda_pp = min(lambda_list);
        lambda_nn = median(lambda_list);
        lambda_pn = max(lambda_list);
    end
    
    lambda_np = (1-gamma)/gamma * (lambda_pn - lambda_nn) + lambda_pp;
    lambda_bn = beta*(alpha-gamma)/(gamma*(alpha-beta)) * lambda_pn + alpha*(gamma - beta)/(gamma*(alpha-beta)) * lambda_nn;
    lambda_bp = (1-alpha)*(gamma-beta)/(gamma*(alpha-beta)) * (lambda_pn - lambda_nn) + lambda_pp;
    
    if lambda_bn > lambda_pp && lambda_bp > lambda_nn
        break;
    end
end
TWD_lambda_matrix = [lambda_pp,lambda_bp,lambda_np;lambda_nn, lambda_bn, lambda_pn];
TWD_Threshold_init = [alpha,beta,gamma];

% ��֤�Ƿ���������
lamda_bp_pp = lambda_bp - lambda_pp;
lamda_bn_nn = lambda_bn - lambda_nn;
lamda_np_bp = lambda_np - lambda_bp;
landa_pn_bn = lambda_pn - lambda_bn;
if lamda_bp_pp * lamda_bn_nn < lamda_np_bp * landa_pn_bn
    fprintf('**************** Yes,ok! it is right ! ! ! **************************');
else
    fprintf('**************** Hello world ! ! ! **************************');
end
end
   

