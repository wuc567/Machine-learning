clc
clear
rng(0)

%%%%%%%%%%%%%%%%%%%%% 序贯阈值参数 %%%%%%%%%%%%%%%%%%%%%
m = 2; n = 3; Hidden_up = 120;
[STWD_lambda_cell, STWD_threshold] = STWD_lambda_matrix(Hidden_up,m,n);

% 保存实验结果
mkdir('F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\Data_TWDSFNN_STWDSFNN');
save('F:\PaperTwo\220504—papertwo-Chinese\Code-Chinese-two\UCI_file\Data_TWDSFNN_STWDSFNN\ESR_STWDSFNN_para_stwd.mat',...
    'STWD_lambda_cell','STWD_threshold')
