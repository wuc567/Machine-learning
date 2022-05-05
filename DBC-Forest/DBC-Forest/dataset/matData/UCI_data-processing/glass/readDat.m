clear
clc

n_patrons= 4177;
n_train = 2089;
fid = fopen('conxuntos.dat','r');
ind_train = fscanf(fid, '%g', n_train);
ind_test = fscanf(fid, '%g',n_patrons);
save('ind_train_test','ind_train','ind_test');

n_train = 3133;
fid1=fopen('conxuntos_kfold.dat','r');
ind_train_1 = fscanf(fid1, '%g', n_train);
ind_test_1 = fscanf(fid, '%g',n_patrons);


