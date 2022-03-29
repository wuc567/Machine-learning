function [X,TX] = selm_kernel(test_data,train_data,train_target,C,kernel_type,kernel_para,Conf)
%%This is second ELM module.

%%%%%%%%%%% Calculate the training output
n = size(train_data,1); 
Omega_train  = kernel_matrix(train_data,kernel_type, kernel_para);
OutputWeight = ((Omega_train+speye(n)/C)\(train_target'*Conf)); 
X = Omega_train * OutputWeight;      % Y: the actual output of the training data

%%%%%%%%%%% Calculate the output of testing input
Omega_test = kernel_matrix(train_data,kernel_type,kernel_para,test_data);
TX = Omega_test' * OutputWeight;     % TY: the actual output of the testing data   


