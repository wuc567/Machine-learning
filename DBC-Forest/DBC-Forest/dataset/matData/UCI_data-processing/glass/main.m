% glass
% author: wuxian, wibsite: https://wuxian.blog.csdn.net
clear;
clc;

data_name = 'glass';
fprintf('lendo problema %s ...\n', data_name);

n_entradas= 9; % 属性数
n_clases= 6; % 类别数
n_patrons(1)= 214; % 数据量（行数）
n_fich= 1;
fich{1}= 'glass.data'; % 文件路径名

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); % 用于存放提取出的属性数据
cl= zeros(n_fich, n_max);             % 用于存放数据标签

n_patrons_total = sum(n_patrons); % 用于显示进度
n_iter=0;

for i_fich=1:n_fich
    f=fopen(fich{i_fich}, 'r'); % 打开文件
    if -1==f
        error('erro en fopen abrindo %s\n', fich{i_fich});
    end
    
    for i=1:n_patrons(i_fich) % 循环对每行数据进行处理
        n_iter=n_iter+1;
        fprintf('%5.1f%%\r', 100*n_iter/n_patrons_total); % 显示处理进度
        
        fscanf(f,'%i',1); % 第一个数字为序号，无需记录
        for j = 1:n_entradas
            temp=fscanf(f, ',%f',1); % 读取下一个数据，以逗号分隔
            x(i_fich,i,j) = temp;    % 保存一个数值到x
        end
        t=fscanf(f,',%i',1);
        if t >= 5  % 原数据标记中没有5，所以后面标号需要-1
            t = t - 1;
        end
        
        cl(i_fich,i) = t - 1;  	% 原标记从1开始，改为从0开始
    end
    fclose(f);% 关闭文件
end


%% 处理完成，保存文件
fprintf('现在保存数据文件...\n')
data = squeeze(x); % 数据
label = cl';% 标签
dataSet = [label,data];
dir_path=['./预处理完成/',data_name];
if exist('./预处理完成/','dir')==0   %该文件夹不存在，则直接创建
    mkdir('./预处理完成/');
end
saveData(dataSet,dir_path); % 保存文件至文件夹
fprintf('预处理完成\n')


%% 数据归一化处理
fprintf('现在进行归一化处理...\n')
min_max_scaling_data = minmax_fun(dataSet(:,2:end), -1, 1);
zscore_normalization_data = zscore(dataSet(:,2:end));
minmax_scaling = [label,min_max_scaling_data];
zscore_normalization = [label, zscore_normalization_data];
if exist('./归一化数据/','dir')==0   %该文件夹不存在，则直接创建
    mkdir('./归一化数据/');
end
save(['./归一化数据/minmax_',data_name,'.mat'],'minmax_scaling');
save(['./归一化数据/zscore_',data_name,'.mat'],'zscore_normalization');
fprintf('归一化完成\n')

%% 划分数据集
train_num = floor(0.7 * n_patrons);
test_num = n_patrons-train_num;
seed = 1;
[TrainSet_minmax,TestSet_minmax, ind_train_minmax, ind_test_minmax]=separate_data(minmax_scaling,train_num,test_num,seed);
[TrainSet_zscore,TestSet_zscore, ind_train_zscore, ind_test_zscore]=separate_data(zscore_normalization,train_num,test_num,seed);

save(['./归一化数据/minmax_Set_',data_name,'.mat'],'TrainSet_minmax','TestSet_minmax');
save(['./归一化数据/zscore_Set_',data_name,'.mat'],'TrainSet_zscore','TestSet_zscore');
fprintf('处理完成\n')