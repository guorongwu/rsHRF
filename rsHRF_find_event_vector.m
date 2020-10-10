function data = rsHRF_find_event_vector(matrix,thr,k,temporal_mask)
%detect events.
N = size(matrix,1);
data = sparse(1,N);
% k = 2;
if isempty(temporal_mask)
    matrix = zscore(matrix);
    for t  = 1+k:N-k
        if matrix(t,1) > thr && all(matrix(t-k:t-1,1)<matrix(t,1)) && all(matrix(t,1)>matrix(t+1:t+k,1))% detects threshold
%         if matrix(t,1) > thr && matrix(t,1) < 3.1 && all(matrix(t-k:t-1,1)<matrix(t,1)) && all(matrix(t,1)>matrix(t+1:t+k,1))% detects threshold
            data(t) = 1 ;
        end
    end
else
    datm = mean(matrix(temporal_mask));
    datstd = std(matrix(temporal_mask));
    datstd(datstd==0) = 1;%in case datstd==0;
    matrix = (matrix-datm)./datstd;
    for t  = 1+k:N-k
        if temporal_mask(t)
            if matrix(t,1) > thr  && all(matrix(t-k:t-1,1)<matrix(t,1)) && all(matrix(t,1)>matrix(t+1:t+k,1))% detects threshold
%             if matrix(t,1) > thr && matrix(t,1) < 3.1 && all(matrix(t-k:t-1,1)<matrix(t,1)) && all(matrix(t,1)>matrix(t+1:t+k,1))% detects threshold
                data(t) = 1 ;
            end
        end
    end
end    

return
