%% adult
% author：wx  website：https://wuxian.blog.csdn.net
clear;
clc;

data_name = 'adult';% 数据集名
fprintf('lendo problema adult...\n');

n_entradas= 14; % 属性数
n_clases= 2; % 分类数
n_fich= 2; % 文件数，含有训练和测试集
fich{1}= 'adult.data';% 训练数据路径
n_patrons(1)= 32561; % 训练集数据量

fich{2}= 'adult.test'; % 测试数据路径
n_patrons(2)= 16281;   % 测试数据量

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); % 属性数据
cl= zeros(n_fich, n_max);             % 标签

discreta = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]; % 1表示该位置的属性需要将字符型离散值转化为数值型

% 字符型离散值的所有取值
workclass = {'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'};
education = {'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'};
marital = {'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'};
occupation = {'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'};
relationship = {'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'};
race = {'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'};
sex = {'Male', 'Female'};
country = {'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'};

% 字符型离散值的所有取值个数
n_workclass=8; 
n_education=16; 
n_marital=7; 
n_occupation=14; 
n_relationship=6; 
n_race=5; 
n_sex=2; 
n_country=41;

for i_fich = 1:n_fich
    f=fopen(fich{i_fich}, 'r');
    if -1==f
        error('打开数据文件出错 %s\n', fich{i_fich});
    end
    
    for i=1:n_patrons(i_fich)
        fprintf('%5.1f%%\r', 100*i/n_patrons(i_fich)); % 显示进度
        
        for j = 1:n_entradas
            if discreta(j)==1
                s = fscanf(f,'%s',1); 
                s = s(1:end-1); % 去掉字符串末尾的逗号
                if strcmp(s, '?')  % 对于缺失值补0
                    x(i_fich,i,j)=0;
                else
                    % 确定具体的属性位置并赋相应变量
                    if j==2
                        n = n_workclass; p=workclass;
                    elseif j==4
                        n = n_education; p=education;
                    elseif j==6
                        n = n_marital; p=marital;
                    elseif j==7
                        n = n_occupation; p=occupation;
                    elseif j==8
                        n = n_relationship; p=relationship;
                    elseif j==9
                        n = n_race; p=race;
                    elseif j==10
                        n = n_sex; p=sex;
                    elseif j==14
                        n = n_country; p=country;
                    end
                    % 根据读取的字符值按排列顺序转化为-1到1之间的分数值
                    a = 2/(n-1); b= (1+n)/(1-n);
                    for k=1:n
                        if strcmp(s, p(k))
                            x(i_fich,i,j) = a*k + b; 
                            break
                        end
                    end
                end
            else % 为0的位置（原数据就是数值型）直接读取原数据
                temp = fscanf(f,'%g',1); 
                x(i_fich,i,j) = temp; 
                fscanf(f,'%c',1);
            end

        end
        
        s = fscanf(f,'%s',1);
        % 将标签转化为数值型（0,1）
        if strcmp(s, '<=50K')||strcmp(s, '<=50K.')
            cl(i_fich,i)=0;
        elseif strcmp(s, '>50K')||strcmp(s, '>50K.')
            cl(i_fich,i)=1;
        else
            error('类别标签 %s 读取出错\n', s)
        end

    end
    fclose(f);
end


%% 处理完成，保存文件
fprintf('现在保存数据文件...\n')
dir_path=['./预处理完成/',data_name];
if exist('./预处理完成/','dir')==0   %该文件夹不存在，则直接创建
    mkdir('./预处理完成/');
end
data_train =  squeeze(x(1,1:n_patrons(1),:)); % 数据
label_train = squeeze(cl(1,1:n_patrons(1)))';% 标签
dataSet_train = [label_train, data_train];
saveData(dataSet_train,[dir_path,'_train']); % 保存文件至文件夹

data_test =  squeeze(x(2,1:n_patrons(2),:)); % 数据
label_test = squeeze(cl(2,1:n_patrons(2)))';% 标签
dataSet_test = [label_test,data_test];
saveData(dataSet_test,[dir_path,'_test']);

fprintf('预处理完成\n')


%% 数据归一化处理
fprintf('现在进行归一化处理...\n')
min_max_scaling_train = minmax_fun(dataSet_train(:,2:end), -1, 1);
min_max_scaling_test = minmax_fun(dataSet_test(:,2:end), -1, 1);

zscore_normalization_train = zscore(dataSet_train(:,2:end));
zscore_normalization_test = zscore(dataSet_test(:,2:end));

% 带标签的数据
minmax_scaling_train = [label_train,min_max_scaling_train];
minmax_scaling_test = [label_test,min_max_scaling_test];

zscore_normalization_train = [label_train, zscore_normalization_train];
zscore_normalization_test = [label_test, zscore_normalization_test];

if exist('./归一化数据/','dir')==0   %该文件夹不存在，则直接创建
    mkdir('./归一化数据/');
end
save(['./归一化数据/minmax_',data_name,'.mat'],'minmax_scaling_train','minmax_scaling_test');
save(['./归一化数据/zscore_',data_name,'.mat'],'zscore_normalization_train','zscore_normalization_test');
fprintf('归一化完成\n');

fprintf('处理完成\n');
