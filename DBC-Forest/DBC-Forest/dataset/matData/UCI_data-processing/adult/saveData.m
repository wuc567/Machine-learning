
function saveData(DataSet,fileName)
% DataSet:����õ����ݼ�
% fileName�����ݼ�������

%% DataΪ����õ����ݼ�����
mat_name = [fileName,'.mat'];
save(mat_name, 'DataSet')  % ����.mat�ļ�
data_name = [fileName,'.data'];
save(data_name,'DataSet','-ASCII'); % ����data�ļ�

% ����txt�ļ�
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