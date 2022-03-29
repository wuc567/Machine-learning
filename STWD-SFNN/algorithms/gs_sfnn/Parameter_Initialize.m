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