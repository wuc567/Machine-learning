
function saveData(DataSet,fileName)
% DataSet:整理好的数据集
% fileName：数据集的名字

%% Data为整理好的数据集矩阵
mat_name = [fileName,'.mat'];
save(mat_name, 'DataSet')  % 保存.mat文件
data_name = [fileName,'.data'];
save(data_name,'DataSet','-ASCII'); % 保存data文件

% 保存txt文件
txt_name = [fileName,'.txt'];
f=fopen(txt_name,'w');
[m,n]=size(DataSet);
for i=1:m
    for j=1:n
        if j==n
            if i~=m
                fprintf(f,'%g \n',DataSet(i,j));
            else
                fprintf(f,'%g',DataSet(i,j));
            end
        else
            fprintf(f,'%g,',DataSet(i,j));
        end
    end
end
fclose(f);

% save iris.txt -ascii Iris
% dlmwrite('iris.txt',Iris);
end