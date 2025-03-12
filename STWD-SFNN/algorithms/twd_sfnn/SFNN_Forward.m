% X_Norm��size(13,8000)
% Y_onehot��size(2,8000)
% Y��size(8000,1)
function [Act_Value,Der_Act_Der_z,Y_logits,Y_prob,Y_hat,Weight_F1,Acc,Kappa,LossFun_Error,LossFun_Value]=SFNN_Forward(X_Norm,Y_onehot,Y,...
                                                                              W1,b1,W2,b2,s,lambda,LossFun,FL_Weight,FL_Adjust)
[Act_Value,Der_Act_Der_z]=Activate_Function(X_Norm,W1,b1,s); %�����,�����size(1,8000)
Y_logits=W2*Act_Value+b2;             % ���������ֵ,size=(�����,������)
Y_logits_max=Y_logits-repmat(max(Y_logits),[size(Y_logits,1),1]); % ��ȥ���е����ֵ,����softmax���
Y_logits_exp=exp(Y_logits_max);
Y_prob=bsxfun(@rdivide,Y_logits_exp,sum(Y_logits_exp,1)); % ��softmax�任��ĸ��ʷֲ�,size=(�����,������)
Y_pred=logical(Y_prob==max(Y_prob));  % argmax����������,size=(�����,������)
Y_hat=vec2ind(Y_pred)';               % ���ȱ���ת���ɱ�ǩ,Y_hat��size=(������,1)
[Weight_F1,Acc,Kappa]=WeightF1_Score(Y,Y_hat);

if LossFun==1  
    %CE��������ʧ���� 
    LossFun_Error=-sum(Y_onehot.*log(Y_prob+10^(-12)),1); %LossFun_Error��size=(1,������), %sum(~,1)��1��ʾ�������
    LossFun_Value=1/size(X_Norm,2).*sum(LossFun_Error)+lambda/2*(sqrt(sum(sum(W1.^2)))+sqrt(sum(sum(W2.^2))));       
else
    %FL�۽���ʧ����
    LossFun_Error=-sum(FL_Weight.*(1-Y_prob).^FL_Adjust.*Y_onehot.*log(Y_prob+10^(-12)),1);  %size=(1,������)
    LossFun_Value=1/size(X_Norm,2).*sum(LossFun_Error)+lambda/2*(sqrt(sum(sum(W1.^2)))+sqrt(sum(sum(W2.^2)))); 
end

function [Weight_F1,Acc,Kappa] = WeightF1_Score(Y,Y_hat)
size_Y=length(Y);
if size_Y==0
    error(message('MATLAB:UNIQUE:NotEnoughInputs'));
end

[ConMat,~] = confusionmat(Y,Y_hat);
sum_column = sum(ConMat,1); % �к�
sum_row = sum(ConMat,2);    % �к�

diag_ConMat = diag(ConMat);
Acc = sum(diag_ConMat)/size_Y;

pe = sum_column * sum_row/(size_Y^2);
Kappa = (Acc-pe)/(1-pe);

if any(sum_column==0) || any(sum_row==0) % �ж��Ƿ���0ֵ
    Weight_F1 = 0;
else
    P = diag_ConMat'./sum_column;
    R = diag_ConMat'./sum_row';
    F1_score = 2*P.*R./(P+R);   % ����ÿ������¶�Ӧ�� F1_Scores
    
    F1_nan = isnan(F1_score);   % �ж��Ƿ��п�ֵ
    if ismember(1,F1_nan)
        [F1_nan_row,F1_nan_column] = find(F1_nan==1);
        F1_score(F1_nan_row,F1_nan_column)=0;
    end   
    
    count=[];
    Y_unique = unique(union(unique(Y),unique(Y_hat)));
    for r = 1:length(Y_unique)
        Y_unique_num = sum(Y==Y_unique(r));
        count = [count,Y_unique_num];  %ÿ����������,1*N
    end
    Weight_F1 = sum(count.*F1_score)/size_Y;  
end
end
end