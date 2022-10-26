load PowerGrid2018.mat
data_feature_mean=mean(data,1); 
data_feature_val=var(data,0,1); 
data_Norm=Normalize(data,data_feature_mean,data_feature_val);
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\PowerGrid2018.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load HTRU2017.mat
data_feature_mean=mean(data,1); 
data_feature_val=var(data,0,1); 
data_Norm=Normalize(data,data_feature_mean,data_feature_val);
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\HTRU2017.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load CreditCard2016.mat
norm_index=[1,12:23];
data_divi=data(:,norm_index);
data_feature_mean=mean(data_divi,1); 
data_feature_val=var(data_divi,0,1);
data_Norm_index=Normalize(data_divi,data_feature_mean,data_feature_val); 
data_Norm=[data_Norm_index(:,1),data(:,2:11),data_Norm_index(:,2:13)];
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\CreditCard2016.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load OnlineNews2015.mat
norm_index=[1:11,18:29];
data_divi=data(:,norm_index);
data_feature_mean=mean(data_divi,1); 
data_feature_val=var(data_divi,0,1);
data_Norm_index=Normalize(data_divi,data_feature_mean,data_feature_val); 
data_Norm=[data_Norm_index(:,11),data(:,12:17),data_Norm_index(:,12:23),data(:,30:58)];
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\OnlineNews2015.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load OnlineIntention2018.mat
norm_index=[1:10];
data_divi=data(:,norm_index);
data_feature_mean=mean(data_divi,1); 
data_feature_val=var(data_divi,0,1);
data_Norm_index=Normalize(data_divi,data_feature_mean,data_feature_val); 
data_Norm=[data_Norm_index(:,1:10),data(:,11:17)];
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\OnlineIntention2018.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Surgical.mat
norm_index=[1,2,14:16,19,23];
data_divi=data(:,norm_index);
data_feature_mean=mean(data_divi,1); 
data_feature_val=var(data_divi,0,1);
data_Norm_index=Normalize(data_divi,data_feature_mean,data_feature_val); 
data_Norm=[data_Norm_index(:,1:2),data(:,3:13),data_Norm_index(:,3:5),data(:,17:18),...
           data_Norm_index(:,6),data(:,20:22),data_Norm_index(:,7),data(:,24)];
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\Surgical.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load SkinSegment2012.mat
% data_Norm=data/255;
% DATA=[data_Norm,label];
% xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\SkinSegment2012.xls',DATA);
% clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Income.mat 
DATA=[data,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\Income.xls',DATA);
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load QSARoraltox2019.mat
% DATA=[data,label];
% xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\QSARoraltox2019.xls',DATA);
% clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Bank.mat
DATA=[data,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\Bank.xls',DATA);
clear 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Diabetic2020.mat
norm_index=[14:16];
data_divi=data(:,norm_index);
data_feature_mean=mean(data_divi,1); 
data_feature_val=var(data_divi,0,1);
data_Norm_index=Normalize(data_divi,data_feature_mean,data_feature_val); 
data_Norm=[data(:,1:13),data_Norm_index(:,1:3),data(:,17:44)];
DATA=[data_Norm,label];
xlswrite('E:\4！Program\2！MatalabCode\ML_Contrast\Diabetic2020.xlsx',DATA);
clear 
