function [beta_hrf, bf, event_bold] = rsHRF_estimation_temporal_basis(data,xBF,temporal_mask,flag_parfor)
% xBF.TR = 2;
% xBF.T = 8;
% xBF.T0 = fix(xBF.T/2); (reference time bin, see slice timing)
% xBF.dt = xBF.TR/xBF.T;
% xBF.AR_lag = 1;
% xBF.thr = 1;
% xBF.len = 25;
% xBF.localK = 2;
% temporal_mask: scrubbing mask.
% By: Guo-Rong Wu (gronwu@gmail.com).
% Faculty of Psychology, Southwest University.
% History:
% - 2015-04-17 - Initial version.
% - 2018-07-18 - fix Warning: Rank deficient, copy code from regress.m
%                       addd AR_lag=0.
%                       set paramater for local peak detection.
% - 2019-08-01 - add temporal basis set (Fourier Set, Gamma function)
if nargin<4
    flag_parfor = 1;
end
    
[N, nvar] = size(data);
if isnan(xBF.order)
    para = rsHRF_global_para;
    xBF.order = para.num_basis;
end
bf = wgr_spm_get_bf(xBF);
warning('off','all')
fprintf('#%d \n',nvar)
beta_hrf = cell(1,nvar);
event_bold= cell(1,nvar);
if flag_parfor
    parfor i=1:nvar
        [beta_hrf{1,i},  event_bold{i}] =wgr_hrf_estimation(data(:,i),xBF,N,bf,temporal_mask);
    end
else
    for i=1:nvar
        [beta_hrf{1,i},  event_bold{i}] =wgr_hrf_estimation(data(:,i),xBF,N,bf,temporal_mask);
    end
end
    
beta_hrf  =cell2mat(beta_hrf);

warning on
return


function [bf] = wgr_spm_get_bf(xBF)
% Fill in basis function structure
% FORMAT [xBF] = spm_get_bf(xBF)
%
% xBF.dt      - time bin length {seconds}
% xBF.name    - description of basis functions specified
%               'Canonical HRF (with time derivative)'
%               'Canonical HRF (with time and dispersion derivatives)'              
%               'Gamma functions'
%               'Fourier set'
%               'Fourier Set (Hanning)'
% xBF.T       - microtime resolution (for 'Canonical HRF*')

% bf      - array of basis functions
%__________________________________________________________________________
%
% spm_get_bf prompts for basis functions to model event or epoch-related
% responses.  The basis functions returned are unitary and orthonormal
% when defined as a function of peri-stimulus time in time-bins.
% It is at this point that the distinction between event and epoch-related 
% responses enters.
%__________________________________________________________________________
% Copyright (C) 1999-2011 Wellcome Trust Centre for Neuroimaging
% Karl Friston
% $Id: spm_get_bf.m 4473 2011-09-08 18:07:45Z guillaume $

% Edited by Guorong Wu. 2019-08-02

%-Length of time bin
%--------------------------------------------------------------------------
dt = xBF.dt; %'time bin for basis functions {secs}';

l  = xBF.len;
h  = xBF.order;
%-Create basis functions
%==========================================================================
%-Microtime resolution
%----------------------------------------------------------------------
fMRI_T  =  xBF.T;

%-Create basis functions
%==========================================================================
fprintf('Basis function: %s\n',xBF.name)
switch xBF.name
 
    case {'Fourier set','Fourier set (Hanning)'}
    %----------------------------------------------------------------------
    pst   = [0:dt:l]';
    pst   = pst/max(pst);
 
    %-Hanning window
    %----------------------------------------------------------------------
    if strcmp(xBF.name,'Fourier set (Hanning)')
        g = (1 - cos(2*pi*pst))/2;
    else
        g = ones(size(pst));
    end
 
    %-Zeroth and higher Fourier terms
    %----------------------------------------------------------------------
    bf    = g;
    for i = 1:h
        bf = [bf g.*sin(i*2*pi*pst)];
        bf = [bf g.*cos(i*2*pi*pst)];   
    end
 
    case {'Gamma functions'}
    %----------------------------------------------------------------------
    pst   = [0:dt:l]';
    bf    = spm_gamma_bf(pst,h);
    otherwise
    %-Canonical hemodynamic response function
    %----------------------------------------------------------------------
    [bf, p]      = spm_hrf(dt,[],fMRI_T);
    p(end) = xBF.len;
    [bf, p]      = spm_hrf(dt,p,fMRI_T);

    %-Add time derivative
    %----------------------------------------------------------------------
    if strfind(xBF.name,'time')

        dp       = 1;
        p(6)     = p(6) + dp;
        D        = (bf(:,1) - spm_hrf(dt,p,fMRI_T))/dp;
        bf       = [bf D(:)];
        p(6)     = p(6) - dp;

        %-Add dispersion derivative
        %------------------------------------------------------------------
        if strfind(xBF.name,'dispersion')

            dp   = 0.01;
            p(3) = p(3) + dp;
            D    = (bf(:,1) - spm_hrf(dt,p,fMRI_T))/dp;
            bf   = [bf D(:)];
        end
    end
end
%-Orthogonalise and fill in basis function structure
%--------------------------------------------------------------------------
bf = spm_orth(bf);
return

%==========================================================================
%- S U B - F U N C T I O N S
%==========================================================================

function bf = spm_gamma_bf(u,h)
% Return basis functions (Gamma functions) used for Volterra expansion
% FORMAT bf = spm_gamma_bf(u,h);
% u   - times {seconds}
% h   - order
% bf  - basis functions (mixture of Gammas)
%__________________________________________________________________________
u     = u(:);
bf    = [];
for i = 2:(1 + h)
        m   = 2^i;
        s   = sqrt(m);
        bf  = [bf spm_Gpdf(u,(m/s)^2,m/s^2)];
end


function [beta_hrf, u0]= wgr_hrf_estimation(dat,xBF,N,bf,temporal_mask)
%% estimate HRF
thr = xBF.thr;
if ~isfield(xBF,'localK')
    if xBF.TR<=2
        localK = 1;
    else
        localK = 2;
    end
else
    localK = xBF.localK;
end
u0 = rsHRF_find_event_vector(dat,thr,localK,temporal_mask);
u = [    full(u0)  
    zeros(xBF.T-1,N) ];
u = reshape(u,1,[]);  %(microtime)
[beta, lag] = wgr_hrf_fit(dat,xBF,u,bf);
beta_hrf = beta; beta_hrf(end+1) = lag;

u0 = find(full(u0(:))); 
return

function  varargout = wgr_hrf_fit(dat,xBF,u,bf)
% u    - BOLD event vector (microtime).
% nlag - time lag from neural event to BOLD event .
lag = xBF.lag;
AR_lag = xBF.AR_lag;
nlag = length(lag);
erm = zeros(1,nlag);
beta = zeros(size(bf,2)+1,nlag);
% fprintf('Converged after ')
for i=1:nlag
    u_lag = [u(1,lag(i)+1:end) zeros(1,lag(i))]';
    [erm(i), beta(:,i)] = wgr_glm_estimation(dat,u_lag,bf,xBF.T,xBF.T0,AR_lag);
end
[~, id] = rsHRF_knee_pt(erm);
if id==nlag
    id = id-1;
end
varargout{1} = beta(:,id+1);
if nargout>1
   varargout{2} = lag(id+1);
end

% fprintf('iterations!\n')
return

function [res_sum, Beta] = wgr_glm_estimation(dat,u,bf,T,T0,AR_lag)
% u: BOLD event vector (microtime).
nscans = size(dat,1);
x = wgr_onset_design(u,bf,T,T0,nscans);
X = [x ones(nscans,1)];
[res_sum, Beta] = wgr_glsco(X,dat,AR_lag);

return

function X = wgr_onset_design(u,bf,T,T0,nscans)
% u: BOLD event vector (microtime).
% bf: - basis set matrix
% T: - microtime resolution (number of time bins per scan)
% T0: - microtime onset (reference time bin, see slice timing)
ind = 1:length(u);
X     = [];
for p = 1:size(bf,2)
    x = conv(u,bf(:,p));
    x = x(ind);
    X = [X x];
end
%-Resample regressors at acquisition times
X = X( (0:(nscans - 1))*T + T0, :);

return

function [res_sum, Beta] = wgr_glsco(X,Y,AR_lag)
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

if nargin < 3;      AR_lag = 0;end
if nargin < 4;   max_iter = 20;end

[nobs, nvar] = size(X);
% Beta = X\Y;
Beta = wgr_regress(Y,X);
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
    Beta = wgr_regress(Y_main,X_main);
    
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

function [hrf,p] = spm_hrf(RT,P,T)
% Haemodynamic response function
% FORMAT [hrf,p] = spm_hrf(RT,p,T)
% RT   - scan repeat time
% p    - parameters of the response function (two Gamma functions)
%
%                                                           defaults
%                                                          {seconds}
%        p(1) - delay of response (relative to onset)          6
%        p(2) - delay of undershoot (relative to onset)       16
%        p(3) - dispersion of response                         1
%        p(4) - dispersion of undershoot                       1
%        p(5) - ratio of response to undershoot                6
%        p(6) - onset {seconds}                                0
%        p(7) - length of kernel {seconds}                    32
%
% T    - microtime resolution [Default: 16]
%
% hrf  - haemodynamic response function
% p    - parameters of the response function
%__________________________________________________________________________
% Copyright (C) 1996-2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_hrf.m 6594 2015-11-06 18:47:05Z guillaume $


%-Parameters of the response function
%--------------------------------------------------------------------------
try
    p = spm_get_defaults('stats.fmri.hrf');
catch
    p = [6 16 1 1 6 0 32];
end
if nargin > 1
    p(1:length(P)) = P;
end

%-Microtime resolution
%--------------------------------------------------------------------------
if nargin > 2
    fMRI_T = T;
else
    fMRI_T = spm_get_defaults('stats.fmri.t');
end

%-Modelled haemodynamic response function - {mixture of Gammas}
%--------------------------------------------------------------------------
dt  = RT/fMRI_T;
u   = [0:ceil(p(7)/dt)] - p(6)/dt;
hrf = spm_Gpdf(u,p(1)/p(3),dt/p(3)) - spm_Gpdf(u,p(2)/p(4),dt/p(4))/p(5);
hrf = hrf([0:floor(p(7)/RT)]*fMRI_T + 1);
hrf = hrf'/sum(hrf);

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
