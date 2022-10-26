%在指定激活函数,并给定权重W1和偏置b1下,计算隐层的激活函数值,同时输出激活函数关于z=w1*X+b1的偏导数

%s：激活函数的类别,前7个是Relu函数,后4个是tanh函数
%X：输入数据集,size=特征数*样本量
%w1：权重,size=隐节数*特征数
%b1：偏置,size=隐节数*样本量

%Activate_Value：激活函数作用后的激活值,size=隐节数*样本量
%Derivative_Activate_Value_Derivative_z:激活函数对z的偏导数,size=隐节数*样本量,其中z=w1*X+b1
%e.g.设置K=1,z的size(K,304)=(1,304),Activate_Value的size(K,304)=(1,304),Der_Activate_Der_z的size(K,304)=(1,304)

function [Activate_Value,Der_Activate_Der_z]=Activate_Function(X,w1,b1,s)
z=w1*X+b1; %size=隐节数*样本量
[r,c]=size(z);
if s==1 %Relu函数
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                Activate_Value(i,j)=0; 
                Der_Activate_Der_z(i,j)=0;
            end
        end
    end
elseif s==2 %Noisy Relu函数
    n=normrnd(0,std(z),1,1);%生成r*c大小的(0,std(z)标准差)范围内的符合正态分布的随机矩阵
    for i=1:r
        for j=1:c
            if z(i,j)>0             
                Activate_Value(i,j)=z(i,j)+n;
                Der_Activate_Der_z(i,j)=1;
            else
                Activate_Value(i,j)=0; 
                Der_Activate_Der_z(i,j)=0;
            end
        end
    end 
elseif s==3  %Leaky Relu函数,神经网络不学习a值
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=0.01;
                Activate_Value(i,j)=a*z(i,j);
                Der_Activate_Der_z(i,j)=a;
            end
        end
    end
elseif s==4  %PRelu函数
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=rand(1,1);%随机生成r*c大小 在(0,1)之间的随机数矩阵,但a不是固定下来的,而是学习得来的
                Activate_Value(i,j)=a.*z(i,j);
                Der_Activate_Der_z(i,j)=a(i,j);
            end
        end
    end
elseif s==5 %RRelu函数
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=normrnd(0,1,1,1);%生成r*c大小的(0,1)范围内的符合正态分布的随机数矩阵
                Activate_Value(i,j)=a.*z(i,j); 
                Der_Activate_Der_z(i,j)=a(i,j);
            end
        end
    end
elseif s==6 %Elu函数,神经网络不学习a值
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=0.2;
                Activate_Value(i,j)=a.*(exp(z(i,j))-1); 
                Der_Activate_Der_z(i,j)=a.*exp(z(i,j));
            end
        end
    end
elseif s==7 %Selu函数
    lambda=1.0507009873554804934193349852946;
    a=1.6732632423543772848170429916717;
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=lambda*z(i,j);
                Der_Activate_Der_z(i,j)=lambda;
            else
                temp=exp(z(i,j));
                Activate_Value(i,j)=lambda*a*(temp-1); 
                Der_Activate_Der_z(i,j)=lambda*a*temp;
            end
        end
    end 
elseif s==8 %tanh函数
    t1=exp(z); 
    t2=exp(-z); 
    Activate_Value=(t1-t2)./(t1+t2); 
    Der_Activate_Der_z=1-Activate_Value.^2;
elseif s==9 %Gelu函数
    tmp1=0.044715.*z.^3+z;
    tmp2=sqrt(2/pi).*(tmp1);
    t1=exp(tmp2);
    t2=exp(-tmp2); 
    tanh1=(t1-t2)./(t1+t2);  %tanh函数
    Activate_Value=0.5.*z.*(1+tanh1); %对应元素相乘
    %%%%%%%%%%% WolfarmAlpha微分 %%%%%%%%%%
    tmp3=0.0356774.*z.^3+0.797885.*z;
    t3=exp(tmp3);
    t4=exp(-tmp3); 
    tanh2=(t3-t4)./(t3+t4); %tanh函数
    sech1=2./(t3+t4);       %sech函数
    sech2=sech1.^2;
    tmp4=0.053516.*z.^3+0.398942.*z;
    Der_Activate_Der_z=0.5.*tanh2+tmp4.*sech2+0.5;
elseif s==10 %Sigmoid函数
    t=exp(-z); 
    Activate_Value=1./(1+t); 
    Der_Activate_Der_z=Activate_Value.*(1-Activate_Value);%对应元素相乘
elseif s==11 %Swish函数=z*sigmoid(beta*z)
    beta=2;
    t1=exp(-beta.*z); 
    t2=1./(1+t1);  %sigmoid(beta*z)
    Activate_Value=z.*t2; 
    Der_Activate_Der_z=t2.*(1+z-Activate_Value);
end
end