clear
clc
warning off
set(0,'DefaultFigureVisible', 'off')

load E:\4��Program\4��Cheng_jiayou\26��STWDNN\data_is\QSAR.mat
Para_Init.data_slide = 1;    % ��Ҫ���ֲ��Լ�
Para_Init.Data_Type  = 7;    % ȫ����һ������
Para_Init.s = 1;   % ���������,ǰ7����Relu����,��4����tanh����
Para_Init.t = 1;   % ���ݵķֲ�����,1�Ƿ��Ӿ��ȷֲ�,���������̬�ֲ�
Para_Init.p = 1;   % ��ͬ�ĵ��η���,1��SGD+Momentum,2��Adam,3�ǲ����������AMSgrad

Para_Init.data_r = size(data,1); % ���ݼ���������
Para_Init.data_c = size(data,2); % ���ݼ���������
Para_Init.ClassNum = numel(unique(label));  % ���ݼ��������Ŀ
label_onehot = full(ind2vec(label',Para_Init.ClassNum))'; % one-hot

%�Ա��㷨֮����ڵ���Ŀ
K_max = 10;
iter = 1;
for k = 1: K_max
    fprintf('##### �����Ŀ ##### = %d\n',iter)
    Para_Contrast_Hidden_GS(data,label,label_onehot,Para_Init,k);
    iter = iter +1;    
end


