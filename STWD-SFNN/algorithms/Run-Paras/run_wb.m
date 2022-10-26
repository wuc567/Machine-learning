clc
clear
rng(0) 

file_path_read = 'C:\Users\lenovo\Desktop\data_is_0911\';
file_name = {'ONP', 'OSP', 'EGSS', 'SE', 'HTRU', ...
                    'DCC', 'SB', 'EOL', 'BM', 'ESR', ...
                    'PCB', 'QSAR', 'OD', 'ROE', 'SSMCR'};  % 处理的文件名称
file_name_per = file_name(5); 
load([file_path_read  char(file_name_per)   '.mat'])
file_path_save = 'F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\data_is_random_0922\';
mkdir(file_path_save)

%%%%%%%%%%%%%%%%%%%%% 数据集的10折划分 %%%%%%%%%%%%%%%%%%%%%
data_r = size(data,1);    % 数据集的样本量
indices_5cv = crossvalind('Kfold',  data_r, 10);
str_list_one = strcat(file_path_save,  char(file_name_per),  '_TWDSFNN_10cv',  '.mat');
save(str_list_one,  'indices_10cv');

%%%%%%%%%%%%%%%%%%%%% 神经网络权重和偏置 %%%%%%%%%%%%%%%%%%%%%
s = 1;
t = 1;
K = 50;
InputDimension = size(data,2);    % 特征集的样本量
ClassNum = numel(unique(label));  % 数据集的类别数目
[Weight_1,bias_1,Weight_2,bias_2]=Parameter_Initialize(s,t,K,InputDimension,ClassNum);
str_list_two = strcat(file_path_save,  char(file_name_per),  '_SFNN_para_',  '11',  '.mat');
save(str_list_two, 'Weight_1','bias_1', 'Weight_2', 'bias_2');

%%%%%%%%%%%%%%%%%%%%% 三支决策的阈值对 %%%%%%%%%%%%%%%%%%%%%
TWD_cases = 1;
[threshold_init,lambda_matrix] = TWD_lambda_paras_v2(TWD_cases);
str_list_three = strcat(file_path_save, char(file_name_per),'_TWDSFNN_para_twd', '.mat');
save(str_list_three, 'threshold_init', 'lambda_matrix');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Weight_1,bias_1,Weight_2,bias_2]=Parameter_Initialize(s,t,K,InputDimension,ClassNum)
%K=1; %K：增加的权重行数,即增加的隐节数目
if s<8 
    if t==1 
        Weight_1=(rand(K,InputDimension)*2-1)*sqrt(6/InputDimension);%Relu函数,服从均匀分布,输入层到隐层
        Weight_2=(rand(ClassNum,K)*2-1)*sqrt(6/K);%Relu函数,服从均匀分布,隐层到输出层
    else
        Weight_1=(normrnd(0,sqrt(2/InputDimension),[K,InputDimension]));%Relu函数,服从正态分布,输入层到隐层
        Weight_2=(normrnd(0,sqrt(2/K),[ClassNum,K]));%Relu函数,服从正态分布,隐层到输出层
    end
else
    if t==1
        Weight_1=(rand(K,InputDimension)*2-1)*sqrt(6/(InputDimension+K));%tanh函数,服从均匀分布,输入层到隐层
        Weight_2=(rand(ClassNum,K)*2-1)*sqrt(6/(K+ClassNum));%tanh函数,服从均匀分布,隐层到输出层
    else
        Weight_1=(normrnd(0,sqrt(2/(InputDimension+K)),[K,InputDimension]));%tanh函数,服从正态分布,输入层到隐层
        Weight_2=(normrnd(0,sqrt(2/(K+ClassNum)),[ClassNum,K]));%tanh函数,服从正态分布,隐层到输出层
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
        lambda_list = randperm(10,2); % 产生10个1到10之间的不重复整数,且返回前3个数
        lambda_nn = min(lambda_list);
        lambda_pn = max(lambda_list);
        
    elseif TWD_cases == 3
        lambda_nn = 0;
        lambda_list = randperm(10,2); % 产生10个1到10之间的不重复整数,且返回前3个数
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
    
    if lambda_bn > lambda_pp && lambda_bp > lambda_nn
        break;
    end
end
TWD_lambda_matrix = [lambda_pp,lambda_bp,lambda_np;lambda_nn, lambda_bn, lambda_pn];
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
end
   

