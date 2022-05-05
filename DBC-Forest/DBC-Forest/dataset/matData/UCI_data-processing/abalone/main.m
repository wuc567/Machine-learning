%% abalone
% author: wuxian, wibsite: https://wuxian.blog.csdn.net
clear;
clc;
data_name = 'abalone';
fprintf(['处理数据集： ',data_name,'abalone 原始数据 ...\n']);
fich= [data_name,'.data'];

n_entradas= 8; % 属性数
n_clases= 3;  % 分类数
n_fich= 1; % 数据集个数
n_patrons= 4177; % 数据量（行数）

x = zeros(n_patrons, n_entradas); % 用于存放提取出的属性数据
cl= zeros(1, n_patrons);% 用于存放数据标签

f=fopen(fich, 'r');% 打开文件
if -1==f
    error('erro en fopen abalone %s\n', fich);
end
for i=1:n_patrons % 循环对每行数据进行处理
    
    fprintf('%5.1f%%\r', 100*i/n_patrons(1));% 显示处理进度
    
    t = fscanf(f, '%c', 1); % 读取一个字符数据
    switch t % 将对应字符替换为数字
        case 'M'
            x(i,1)=-1;
        case 'F'
            x(i,1)=0;
        case 'I'
            x(i,1)=1;
    end
    
    for j=2:n_entradas
        fscanf(f,'%c',1); % 中间有分隔符，后移1个位置
        x(i,j) = fscanf(f,'%f', 1);% 依次读取这一行所有属性
    end
    
    fscanf(f,'%c',1); 
    t = fscanf(f,'%i', 1); % 读取最后的标记值
    % 根据范围将连续的标记值离散化为三类
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

%% 处理完成，保存文件
fprintf('现在保存数据文件...\n')
data = x; % 数据
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