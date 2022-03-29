clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load E:\4—Program\4—Cheng_jiayou\26—STWDNN\data_is\QSAR.mat
Para_Init.data_r = size(data,1); % 数据集的样本量
Para_Init.data_c = size(data,2); % 数据集的特征数
Para_Init.ClassNum = numel(unique(label));  % 数据集的类别数目
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

%对比算法之隐层节点数目
Para_Init.eta = 2; %介于1~10之间
K1 = ceil(sqrt(Para_Init.data_c + Para_Init.ClassNum+Para_Init.eta));
K2 = ceil(log2(Para_Init.data_c));
K3 = ceil(sqrt(Para_Init.data_c * Para_Init.ClassNum));

iter = 1;
for k = [K1,K2,K3]
    fprintf('##### 对比算法 ##### = %d\n',iter)
    Para_Contrast_Hidden_K123(data,label,label_onehot,Para_Init,k,iter);
    iter = iter +1;    
end


