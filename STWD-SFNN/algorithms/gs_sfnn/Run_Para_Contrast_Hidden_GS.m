clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load E:\4—Program\4—Cheng_jiayou\26—STWDNN\data_is\QSAR.mat
Para_Init.data_slide = 1;    % 需要划分测试集
Para_Init.Data_Type  = 7;    % 全部归一化处理
Para_Init.s = 1;   % 激活函数类型,前7个是Relu函数,后4个是tanh函数
Para_Init.t = 1;   % 数据的分布类型,1是服从均匀分布,否则服从正态分布
Para_Init.p = 1;   % 不同的调参方法,1是SGD+Momentum,2是Adam,3是不带修正项的AMSgrad

Para_Init.data_r = size(data,1); % 数据集的样本量
Para_Init.data_c = size(data,2); % 数据集的特征数
Para_Init.ClassNum = numel(unique(label));  % 数据集的类别数目
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

%对比算法之隐层节点数目
K_max = 10;
iter = 1;
for k = 1: K_max
    fprintf('##### 结点数目 ##### = %d\n',iter)
    Para_Contrast_Hidden_GS(data,label,label_onehot,Para_Init,k);
    iter = iter +1;    
end


