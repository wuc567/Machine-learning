% glass
% author: wuxian, wibsite: https://wuxian.blog.csdn.net
clear;
clc;

data_name = 'glass';
fprintf('lendo problema %s ...\n', data_name);

n_entradas= 9; % ������
n_clases= 6; % �����
n_patrons(1)= 214; % ��������������
n_fich= 1;
fich{1}= 'glass.data'; % �ļ�·����

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); % ���ڴ����ȡ������������
cl= zeros(n_fich, n_max);             % ���ڴ�����ݱ�ǩ

n_patrons_total = sum(n_patrons); % ������ʾ����
n_iter=0;

for i_fich=1:n_fich
    f=fopen(fich{i_fich}, 'r'); % ���ļ�
    if -1==f
        error('erro en fopen abrindo %s\n', fich{i_fich});
    end
    
    for i=1:n_patrons(i_fich) % ѭ����ÿ�����ݽ��д���
        n_iter=n_iter+1;
        fprintf('%5.1f%%\r', 100*n_iter/n_patrons_total); % ��ʾ�������
        
        fscanf(f,'%i',1); % ��һ������Ϊ��ţ������¼
        for j = 1:n_entradas
            temp=fscanf(f, ',%f',1); % ��ȡ��һ�����ݣ��Զ��ŷָ�
            x(i_fich,i,j) = temp;    % ����һ����ֵ��x
        end
        t=fscanf(f,',%i',1);
        if t >= 5  % ԭ���ݱ����û��5�����Ժ�������Ҫ-1
            t = t - 1;
        end
        
        cl(i_fich,i) = t - 1;  	% ԭ��Ǵ�1��ʼ����Ϊ��0��ʼ
    end
    fclose(f);% �ر��ļ�
end


%% ������ɣ������ļ�
fprintf('���ڱ��������ļ�...\n')
data = squeeze(x); % ����
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