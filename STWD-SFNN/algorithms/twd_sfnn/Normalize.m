function data_normalize=Normalize(data,data_feature_mean,data_feature_val)
[data_r,data_c]=size(data);     %data_r=391,data_c=16
data_normalize=[];epsilon=10^(-8);
for j=1:data_c
    for i=1:data_r
        data_normalize(i,j)=(data(i,j)-data_feature_mean(j))/(data_feature_val(j)+epsilon);
    end
end