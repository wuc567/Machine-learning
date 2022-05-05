function [data]=minmax_fun(dataSet,xmin,xmax)
    scale=xmax-xmin;
    data = dataSet;
    % πÈ“ªªØ
    min_data=min(dataSet);
    max_data=max(dataSet);
    
    for i=1:length(min_data)
        diff=(max_data(1,i)-min_data(1,i));
        if diff~=0
            data(:,i)=scale*(dataSet(:,i)-min_data(1,i))./diff + xmin;
        else
            data(:,i)=zeros(size(dataSet(:,i)));
        end
    end
    
end