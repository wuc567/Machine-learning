%% adult
% author��wx  website��https://wuxian.blog.csdn.net
clear;
clc;

data_name = 'adult';% ���ݼ���
fprintf('lendo problema adult...\n');

n_entradas= 14; % ������
n_clases= 2; % ������
n_fich= 2; % �ļ���������ѵ���Ͳ��Լ�
fich{1}= 'adult.data';% ѵ������·��
n_patrons(1)= 32561; % ѵ����������

fich{2}= 'adult.test'; % ��������·��
n_patrons(2)= 16281;   % ����������

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); % ��������
cl= zeros(n_fich, n_max);             % ��ǩ

discreta = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]; % 1��ʾ��λ�õ�������Ҫ���ַ�����ɢֵת��Ϊ��ֵ��

% �ַ�����ɢֵ������ȡֵ
workclass = {'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'};
education = {'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'};
marital = {'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'};
occupation = {'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'};
relationship = {'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'};
race = {'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'};
sex = {'Male', 'Female'};
country = {'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'};

% �ַ�����ɢֵ������ȡֵ����
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
        error('�������ļ����� %s\n', fich{i_fich});
    end
    
    for i=1:n_patrons(i_fich)
        fprintf('%5.1f%%\r', 100*i/n_patrons(i_fich)); % ��ʾ����
        
        for j = 1:n_entradas
            if discreta(j)==1
                s = fscanf(f,'%s',1); 
                s = s(1:end-1); % ȥ���ַ���ĩβ�Ķ���
                if strcmp(s, '?')  % ����ȱʧֵ��0
                    x(i_fich,i,j)=0;
                else
                    % ȷ�����������λ�ò�����Ӧ����
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
                    % ���ݶ�ȡ���ַ�ֵ������˳��ת��Ϊ-1��1֮��ķ���ֵ
                    a = 2/(n-1); b= (1+n)/(1-n);
                    for k=1:n
                        if strcmp(s, p(k))
                            x(i_fich,i,j) = a*k + b; 
                            break
                        end
                    end
                end
            else % Ϊ0��λ�ã�ԭ���ݾ�����ֵ�ͣ�ֱ�Ӷ�ȡԭ����
                temp = fscanf(f,'%g',1); 
                x(i_fich,i,j) = temp; 
                fscanf(f,'%c',1);
            end

        end
        
        s = fscanf(f,'%s',1);
        % ����ǩת��Ϊ��ֵ�ͣ�0,1��
        if strcmp(s, '<=50K')||strcmp(s, '<=50K.')
            cl(i_fich,i)=0;
        elseif strcmp(s, '>50K')||strcmp(s, '>50K.')
            cl(i_fich,i)=1;
        else
            error('����ǩ %s ��ȡ����\n', s)
        end

    end
    fclose(f);
end


%% ������ɣ������ļ�
fprintf('���ڱ��������ļ�...\n')
dir_path=['./Ԥ�������/',data_name];
if exist('./Ԥ�������/','dir')==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir('./Ԥ�������/');
end
data_train =  squeeze(x(1,1:n_patrons(1),:)); % ����
label_train = squeeze(cl(1,1:n_patrons(1)))';% ��ǩ
dataSet_train = [label_train, data_train];
saveData(dataSet_train,[dir_path,'_train']); % �����ļ����ļ���

data_test =  squeeze(x(2,1:n_patrons(2),:)); % ����
label_test = squeeze(cl(2,1:n_patrons(2)))';% ��ǩ
dataSet_test = [label_test,data_test];
saveData(dataSet_test,[dir_path,'_test']);

fprintf('Ԥ�������\n')


%% ���ݹ�һ������
fprintf('���ڽ��й�һ������...\n')
min_max_scaling_train = minmax_fun(dataSet_train(:,2:end), -1, 1);
min_max_scaling_test = minmax_fun(dataSet_test(:,2:end), -1, 1);

zscore_normalization_train = zscore(dataSet_train(:,2:end));
zscore_normalization_test = zscore(dataSet_test(:,2:end));

% ����ǩ������
minmax_scaling_train = [label_train,min_max_scaling_train];
minmax_scaling_test = [label_test,min_max_scaling_test];

zscore_normalization_train = [label_train, zscore_normalization_train];
zscore_normalization_test = [label_test, zscore_normalization_test];

if exist('./��һ������/','dir')==0   %���ļ��в����ڣ���ֱ�Ӵ���
    mkdir('./��һ������/');
end
save(['./��һ������/minmax_',data_name,'.mat'],'minmax_scaling_train','minmax_scaling_test');
save(['./��һ������/zscore_',data_name,'.mat'],'zscore_normalization_train','zscore_normalization_test');
fprintf('��һ�����\n');

fprintf('�������\n');
