%% abalone
% author: wuxian, wibsite: https://wuxian.blog.csdn.net
clear;
clc;
data_name = 'abalone';
fprintf(['�������ݼ��� ',data_name,'abalone ԭʼ���� ...\n']);
fich= [data_name,'.data'];

n_entradas= 8; % ������
n_clases= 3;  % ������
n_fich= 1; % ���ݼ�����
n_patrons= 4177; % ��������������

x = zeros(n_patrons, n_entradas); % ���ڴ����ȡ������������
cl= zeros(1, n_patrons);% ���ڴ�����ݱ�ǩ

f=fopen(fich, 'r');% ���ļ�
if -1==f
    error('erro en fopen abalone %s\n', fich);
end
for i=1:n_patrons % ѭ����ÿ�����ݽ��д���
    
    fprintf('%5.1f%%\r', 100*i/n_patrons(1));% ��ʾ�������
    
    t = fscanf(f, '%c', 1); % ��ȡһ���ַ�����
    switch t % ����Ӧ�ַ��滻Ϊ����
        case 'M'
            x(i,1)=-1;
        case 'F'
            x(i,1)=0;
        case 'I'
            x(i,1)=1;
    end
    
    for j=2:n_entradas
        fscanf(f,'%c',1); % �м��зָ���������1��λ��
        x(i,j) = fscanf(f,'%f', 1);% ���ζ�ȡ��һ����������
    end
    
    fscanf(f,'%c',1); 
    t = fscanf(f,'%i', 1); % ��ȡ���ı��ֵ
    % ���ݷ�Χ�������ı��ֵ��ɢ��Ϊ����
    if t < 9
        cl(1,i)=0;
    elseif t < 11
        cl(1,i)=1;
    else
        cl(1,i)=2;
    end
    fscanf(f,'%c',1);
    
end
fclose(f);

%% ������ɣ������ļ�
fprintf('���ڱ��������ļ�...\n')
data = x; % ����
label = cl';% ��ǩ
dataSet = [label,data];
dir_path=['./Ԥ�������/',data_name];
if exist('./Ԥ�������/','dir')==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir('./Ԥ�������/');
end
saveData(dataSet,dir_path); % �����ļ����ļ���
fprintf('Ԥ�������\n')


%% ���ݹ�һ������
fprintf('���ڽ��й�һ������...\n')
min_max_scaling_data = minmax_fun(dataSet(:,2:end), -1, 1);
zscore_normalization_data = zscore(dataSet(:,2:end));
minmax_scaling = [label,min_max_scaling_data];
zscore_normalization = [label, zscore_normalization_data];
if exist('./��һ������/','dir')==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir('./��һ������/');
end
save(['./��һ������/minmax_',data_name,'.mat'],'minmax_scaling');
save(['./��һ������/zscore_',data_name,'.mat'],'zscore_normalization');
fprintf('��һ�����\n')

%% �������ݼ�
train_num = floor(0.7 * n_patrons);
test_num = n_patrons-train_num;
seed = 1;
[TrainSet_minmax,TestSet_minmax, ind_train_minmax, ind_test_minmax]=separate_data(minmax_scaling,train_num,test_num,seed);
[TrainSet_zscore,TestSet_zscore, ind_train_zscore, ind_test_zscore]=separate_data(zscore_normalization,train_num,test_num,seed);

save(['./��һ������/minmax_Set_',data_name,'.mat'],'TrainSet_minmax','TestSet_minmax');
save(['./��һ������/zscore_Set_',data_name,'.mat'],'TrainSet_zscore','TestSet_zscore');
fprintf('�������\n')