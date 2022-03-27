%��ָ�������,������Ȩ��W1��ƫ��b1��,��������ļ����ֵ,ͬʱ������������z=w1*X+b1��ƫ����

%s������������,ǰ7����Relu����,��4����tanh����
%X���������ݼ�,size=������*������
%w1��Ȩ��,size=������*������
%b1��ƫ��,size=������*������

%Activate_Value����������ú�ļ���ֵ,size=������*������
%Derivative_Activate_Value_Derivative_z:�������z��ƫ����,size=������*������,����z=w1*X+b1
%e.g.����K=1,z��size(K,304)=(1,304),Activate_Value��size(K,304)=(1,304),Der_Activate_Der_z��size(K,304)=(1,304)

function [Activate_Value,Der_Activate_Der_z]=Activate_Function(X,w1,b1,s)
z=w1*X+b1; %size=������*������
[r,c]=size(z);
if s==1 %Relu����
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
elseif s==2 %Noisy Relu����
    n=normrnd(0,std(z),1,1);%����r*c��С��(0,std(z)��׼��)��Χ�ڵķ�����̬�ֲ����������
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
elseif s==3  %Leaky Relu����,�����粻ѧϰaֵ
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
elseif s==4  %PRelu����
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=rand(1,1);%�������r*c��С ��(0,1)֮������������,��a���ǹ̶�������,����ѧϰ������
                Activate_Value(i,j)=a.*z(i,j);
                Der_Activate_Der_z(i,j)=a(i,j);
            end
        end
    end
elseif s==5 %RRelu����
    for i=1:r
        for j=1:c
            if z(i,j)>0
                Activate_Value(i,j)=z(i,j);
                Der_Activate_Der_z(i,j)=1;
            else
                a=normrnd(0,1,1,1);%����r*c��С��(0,1)��Χ�ڵķ�����̬�ֲ������������
                Activate_Value(i,j)=a.*z(i,j); 
                Der_Activate_Der_z(i,j)=a(i,j);
            end
        end
    end
elseif s==6 %Elu����,�����粻ѧϰaֵ
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
elseif s==7 %Selu����
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
elseif s==8 %tanh����
    t1=exp(z); 
    t2=exp(-z); 
    Activate_Value=(t1-t2)./(t1+t2); 
    Der_Activate_Der_z=1-Activate_Value.^2;
elseif s==9 %Gelu����
    tmp1=0.044715.*z.^3+z;
    tmp2=sqrt(2/pi).*(tmp1);
    t1=exp(tmp2);
    t2=exp(-tmp2); 
    tanh1=(t1-t2)./(t1+t2);  %tanh����
    Activate_Value=0.5.*z.*(1+tanh1); %��ӦԪ�����
    %%%%%%%%%%% WolfarmAlpha΢�� %%%%%%%%%%
    tmp3=0.0356774.*z.^3+0.797885.*z;
    t3=exp(tmp3);
    t4=exp(-tmp3); 
    tanh2=(t3-t4)./(t3+t4); %tanh����
    sech1=2./(t3+t4);       %sech����
    sech2=sech1.^2;
    tmp4=0.053516.*z.^3+0.398942.*z;
    Der_Activate_Der_z=0.5.*tanh2+tmp4.*sech2+0.5;
elseif s==10 %Sigmoid����
    t=exp(-z); 
    Activate_Value=1./(1+t); 
    Der_Activate_Der_z=Activate_Value.*(1-Activate_Value);%��ӦԪ�����
elseif s==11 %Swish����=z*sigmoid(beta*z)
    beta=2;
    t1=exp(-beta.*z); 
    t2=1./(1+t1);  %sigmoid(beta*z)
    Activate_Value=z.*t2; 
    Der_Activate_Der_z=t2.*(1+z-Activate_Value);
end
end