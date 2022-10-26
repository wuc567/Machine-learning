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