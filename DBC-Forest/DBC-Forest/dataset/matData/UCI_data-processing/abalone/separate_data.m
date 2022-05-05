function [TrainSet,TestSet, ind_train, ind_test]=separate_data(data, train_num, test_num, seed)

%     data=load(Data_File); % �����ļ�����

    num_data=size(data,1); % ��������
    if num_data~=train_num+test_num
        error('��������һ��');
    end
    
    % ѵ���Ͳ������ݼ��ı���
    rand('state',seed); % �������������
    ratio_train_test=train_num/(train_num+test_num);
    
    label=data(:,1);
    [classType, i_label, i_class] = unique(label); % ȡ����ͬ�ı�ǩ����ͬ��
    num_class=length(classType);
    class_data_train=[];
    class_data_test=[];
    % ��ȡÿһ������ݵ�����
%     num_train_class = zeros(1,num_class);
%     num_test_class = zeros(1,num_class);
    for i=1:num_class
        ind_class=find(label==classType(i));
        class_data=data(ind_class,:);
        % ����train_num, test_num�ı������ѡ��ÿ�������
        len_ind_class=length(ind_class);
        len_ind_train=round(len_ind_class*ratio_train_test);
        
        ind_class=randperm(len_ind_class)';
        ind_train=ind_class(1:len_ind_train);
        ind_test=ind_class(len_ind_train+1:end);
        
        class_data_train=[class_data_train;class_data(ind_train,:)];
        class_data_test=[class_data_test;class_data(ind_test,:)];
        
%         num_train_class(i) = length(class_data(ind_train,:));
%         num_test_class(i) = length(class_data(ind_test,:));
    end
    
    % �Ƚ�ɸѡ��ѵ�����Լ���Ŀ�Ƿ���Ԥ��һ��
    if size(class_data_train,1) > train_num
        class_data_test = [class_data_test;class_data_train(train_num+1:end,:)];
        class_data_train = class_data_train(1:train_num,:);
    else
        num=train_num-size(class_data_train,1);
        class_data_train = [class_data_train;class_data_test(end-num+1:end,:)];
        class_data_test = class_data_test(1:end-num,:);
        
    end
    
    
%     TestSet.T=class_data_test(:,1);
    TestSet=class_data_test(:,1:end);
    
%     TrainSet.T=class_data_train(:,1);
    TrainSet=class_data_train(:,1:end);
    
   
        
end