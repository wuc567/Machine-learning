clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load E:\4��Program\4��Cheng_jiayou\26��STWDNN\data_is\QSAR.mat
Para_Init.data_r = size(data,1); % ���ݼ���������
Para_Init.data_c = size(data,2); % ���ݼ���������
Para_Init.ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

%�Ա��㷨֮����ڵ���Ŀ
Para_Init.eta = 2; %����1~10֮��
K1 = ceil(sqrt(Para_Init.data_c + Para_Init.ClassNum+Para_Init.eta));
K2 = ceil(log2(Para_Init.data_c));
K3 = ceil(sqrt(Para_Init.data_c * Para_Init.ClassNum));

iter = 1;
for k = [K1,K2,K3]
    fprintf('##### �Ա��㷨 ##### = %d\n',iter)
    Para_Contrast_Hidden_K123(data,label,label_onehot,Para_Init,k,iter);
    iter = iter +1;    
end


