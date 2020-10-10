function [beta_rshrf,event_bold] = rsHRF_estimation_FIR(data,para,temporal_mask)
% temporal_mask: generated from scrubbing.
% By: Guo-Rong Wu (gronwu@gmail.com).
% Faculty of Psychology, Southwest University.
% History:
% - 2015-04-17 - Initial version.

para.temporal_mask=temporal_mask;
[N,nvar] = size(data);

% warning off
beta_rshrf = cell(1,nvar);
event_bold= cell(1,nvar);
parfor i=1:nvar
    [beta_rshrf{i}, event_bold{i}] = wgr_FIR_estimation_HRF(data(:,i),para,N);
end
beta_rshrf  =cell2mat(beta_rshrf);
% warning on
return

function [beta_hrf,u] = wgr_FIR_estimation_HRF(data,para,N)
if ~isfield(para,'localK')
    if para.TR<=2
        localK = 1;
    else
        localK = 2;
    end
else
    localK = para.localK;
end
u = rsHRF_find_event_vector(data,para.thr,localK,para.temporal_mask);
u = find(full(u(:)));
lag = para.lag;
nlag = length(lag);
len_bin=floor(para.len/para.TR);
hrf = zeros(len_bin+1,nlag);
Cov_E = zeros(1,nlag);
kk=1;
for i_lag=1:nlag
    RR = u-i_lag; RR(RR<=0)=[];
    if ~isempty(RR)
        design = zeros(N,1);
        design(RR) = 1;
        [hrf(:,kk),Cov_E(kk)] = wgr_Fit_FIR2(design,data,para);
    else
        Cov_E(kk) = inf;
    end
    kk = kk+1;
end
[~, ind] = rsHRF_knee_pt(Cov_E); % this is a function to find the elbow point of a curve, there should be an equivalent in Python
if ind==length(Cov_E)
    ind = ind-1;
end
beta_hrf = hrf(:,ind+1);
beta_hrf(end+1) = lag(ind+1);

function [hrf,res_sum] = wgr_Fit_FIR2(input,output,para)
% NN : length of impulse response
NN = floor(para.len/para.TR);
flag_sfir = double(strcmp(para.estimation,'sFIR'));
% X : convolution matrix
X = toeplitz(input, [input(1) zeros(1, NN-1)]); 
X = [X ones(size(input))];
% h : impulse response estimate
if flag_sfir
    MRI = wgr_smFIR(NN-1,para.TR); 
    sigma = 1; sMRI0 = sigma^2*MRI;
    sMRI = zeros(NN+1);
    sMRI(1:NN,1:NN) = sMRI0; 
    if ~para.AR_lag
        hrf = (X'*X+sMRI)\X'*output; %smoothed solution
        resid = output - X * hrf;
        res_sum=cov(resid);
    else
        [res_sum, hrf] = wgr_glsco_sm(X,output,para.AR_lag,sMRI);
    end
else
    
    if ~para.AR_lag
        hrf = X \ output;  
        resid = output - X * hrf;
        res_sum=cov(resid);
    else
        [res_sum, hrf] = wgr_glsco_sm(X,output,para.AR_lag,[]);
    end
end


function MRI = wgr_smFIR(nh,dt)
fwhm =7; % fwhm=7 seconds smoothing - ref. Goutte
C=(1:nh)'*(ones(1,nh));
h = sqrt(1/(fwhm/dt));       
v = 0.1;
R = v*exp(-h/2*(C-C').^2);
RI = inv(R);
MRI = zeros(nh+1);
MRI(1:nh,1:nh) = RI;
return

function [res_sum, Beta] = wgr_glsco_sm(X,Y,AR_lag,sMRI,max_iter)
% Linear regression when disturbance terms follow AR(p)
% -----------------------------------
% Model:
% Yt = Xt * Beta + ut , 
% ut = Phi1 * u(t-1) + ... + Phip * u(t-p) + et
% where et ~ N(0,s^2)
% -----------------------------------
% Algorithm: 
% Cochrane-Orcutt iterated regression (Feasible generalized least squares)
% -----------------------------------
% Usage:
% Y = dependent variable (n * 1 vector)
% X = regressors (n * k matrix)
% AR_lag = number of lags in AR process
% -----------------------------------
% Returns:
% Beta = estimator corresponding to the k regressors 

if nargin < 3;   AR_lag = 1; sMRI = []; end
if nargin < 5;   max_iter = 20;end

[nobs] = size(X,1);
if isempty(sMRI)
    Beta = wgr_regress(Y,X);
else
    Beta = (X'*X+sMRI)\X'*Y; %smoothed solution
end
resid = Y - X * Beta;
if ~AR_lag
    %res_sum = sum(resid.^2);
    res_sum=cov(resid);
    return
end
max_tol = min(1e-6,max(abs(Beta))/1000);
for r = 1:max_iter    
%     fprintf('Iteration No. %d\n',r)   
    Beta_temp = Beta;

    X_AR = zeros(nobs-2*AR_lag,AR_lag);
    for m = 1:AR_lag
        X_AR(:,m) = resid(AR_lag+1-m:nobs-AR_lag-m);
    end
    
    Y_AR = resid(AR_lag+1:nobs-AR_lag);
%     AR_para = X_AR\Y_AR;
    AR_para = wgr_regress(Y_AR,X_AR);
    
    X_main = X(AR_lag+1:nobs,:);
    Y_main = Y(AR_lag+1:nobs);
    for m = 1:AR_lag
        X_main = X_main-AR_para(m)*X(AR_lag+1-m:nobs-m,:);
        Y_main = Y_main-AR_para(m)*Y(AR_lag+1-m:nobs-m);
    end
    
%     Beta = X_main\Y_main;
    if isempty(sMRI)
        Beta = wgr_regress(Y_main,X_main);
    else
        Beta = (X_main'*X_main+sMRI)\X_main'*Y_main; %smoothed solution    
    end
    
    resid = Y(AR_lag+1:nobs) - X(AR_lag+1:nobs,:)*Beta;
    if max(abs(Beta-Beta_temp)) < max_tol
%         fprintf('%d ,',r) 
%         fprintf('Converged after %d iterations!\n',r) 
        break
    end    
    
end
% res_sum = sum(resid.^2);
res_sum = cov(resid);
%if r == max_iter
%    fprintf('Maximum %d iteration reached.\n',max_iter)    
%end
return

function b = wgr_regress(y,X)
% copy from function [b,bint,r,rint,stats] = regress(y,X,alpha)
[n,ncolX] = size(X);
% Use the rank-revealing QR to remove dependent columns of X.
[Q,R,perm] = qr(X,0);
if isempty(R)
    p = 0;
elseif isvector(R)
    p = double(abs(R(1))>0);
else
    p = sum(abs(diag(R)) > max(n,ncolX)*eps(R(1)));
end
if p < ncolX
%     warning(message('stats:regress:RankDefDesignMat'));
    R = R(1:p,1:p);
    Q = Q(:,1:p);
    perm = perm(1:p);
end

% Compute the LS coefficients, filling in zeros in elements corresponding
% to rows of X that were thrown out.
b = zeros(ncolX,1);
b(perm) = R \ (Q'*y);
