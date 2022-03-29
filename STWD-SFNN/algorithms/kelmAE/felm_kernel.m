function [X,TX] = felm_kernel(test_data,newtrain_data,train_data, C, kernel_type, kernel_para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is first ELM module.

%%%%%%%%%%% Calculate the OutputWeight
n = size(train_data,1); 
Omega_train = kernel_matrix(newtrain_data,kernel_type, kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(train_data)); 
X = Omega_train * OutputWeight;   % Y: the actual output of the training data

%%%%%%%%%%% Calculate the output of testing input
Omega_test = kernel_matrix(train_data,kernel_type,kernel_para,test_data);
TX = Omega_test'*OutputWeight;    % TY: the actual output of the testing data

 
    

    




