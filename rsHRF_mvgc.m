function [F,pvalue] = rsHRF_mvgc(data,order,regmode,flag_1to2,flag_pvalue)
% Calculate time-domain multivariate Granger causalities
% data: nvars x nobs x ntrials
% F(i,j): from i to j.
% pvalue: F-test.
% 
%% References
% [1] L. Barnett and A. K. Seth, The MVGC The MVGC multivariate Granger causality toolbox: 
%     A new approach to Granger-causal inference. J. Neurosci. Methods 223, 2014
% [2] X. Kitagawa, An algorithm for solving the matrix equation X = F X F'+S,
%     Internat. J. Control 25(5), 1977.
% [3] S. Hammarling, "Numerical solution of the stable, non-negative definite
%     Lyapunov equation", IMA J. Numer. Anal. 2, 1982.

[nvars,nobs,ntrials] = size(data);
% full regression
% owstate = warn_supp;
[G,SIG,info] = data_to_autocov(data,order,regmode);
% warn_test(owstate,    'in full regression - bad autocovariance matrix? Check output of ''var_info''');
% if warn_if(isbad(SIG),'in full regression - regression failed'), return; end % show-stopper!

if ~flag_1to2
    F = nan(nvars);
    pvalue = nan(nvars);
else
    F = NaN;
    pvalue = nan;
end
% var_info(info,true); % report results (and bail out on error)
if info.error
    fprintf('.')
else
    LSIG = log(diag(SIG));
    for j = 1:nvars
        % reduced regression
        jo = [1:j-1 j+1:nvars]; % omit j
%         owstate = warn_supp;
        [~,SIGj] = autocov_to_var(G(jo,jo,:));
%         warn_test(owstate,     sprintf('in reduced regression for target node %d - bad autocovariance matrix? Check output of ''var_info''',j));
%         if warn_if(isbad(SIGj),sprintf('in reduced regression for target node %d - regression failed',j)), continue; end % show-stopper!

        LSIGj = log(diag(SIGj));
        if flag_1to2
            F = LSIGj(1)-LSIG(2); break
        else
            for ii=1:nvars-1
                i = jo(ii);
                F(j,i) = LSIGj(ii)-LSIG(i);
            end
        end
    end
    if flag_pvalue
        tstat     = '';
        pvalue = mvgc_pval(F,order,nobs,ntrials,1,1,nvars-2,tstat); % take careful note of arguments!
    end
end

function [G,SIG,info] = data_to_autocov(X,p,regmode)

if nargin < 3 || isempty(regmode), regmode = 'LWR'; end

[n,m,N] = size(X);
assert(p < m,'too many lags');
p1 = p+1;

A   = NaN; 
SIG = NaN; 
G = NaN;
info.error = 1;

X = demean(X); % no constant term

if  strcmpi(regmode,'OLS') % OLS (QR decomposition)

    M = N*(m-p);
    np = n*p;

    % stack lags

    X0 = reshape(X(:,p1:m,:),n,M); % concatenate trials for unlagged observations
    XL = zeros(n,p,M);
    for k = 1:p
        XL(:,k,:) = reshape(X(:,p1-k:m-k,:),n,M); % concatenate trials for k-lagged observations
    end
    XL = reshape(XL,np,M);         % stack lags

    A = X0/XL;                     % OLS using QR decomposition
    if isbad(A); return; end       % something went badly wrong

    if nargout > 1
        E   = X0-A*XL;             % residuals
        SIG = (E*E')/(M-1);        % residuals covariance matrix
        E   = reshape(E,n,m-p,N);  % put residuals back into per-trial form
    end
    
    A = reshape(A,n,n,p);          % so A(:,:,k) is the k-lag coefficients matrix

elseif strcmpi(regmode,'LWR') % LWR (Morf)

    q1n = p1*n;

    I = eye(n);

    % store lags

    XX = zeros(n,p1,m+p,N);
    for k = 0:p
        XX(:,k+1,k+1:k+m,:) = X; % k-lagged observations
    end

    % initialise recursion

    AF = zeros(n,q1n); % forward  AR coefficients
    AB = zeros(n,q1n); % backward AR coefficients (reversed compared with Morf's treatment)

    k  = 1;            % model order is k-1
    kn = k*n;
    M  = N*(m-k);
    kf = 1:kn;         % forward  indices
    kb = q1n-kn+1:q1n; % backward indices

    XF = reshape(XX(:,1:k,k+1:m,:),kn,M);
    XB = reshape(XX(:,1:k,k:m-1,:),kn,M);

    [CXF,cholp] = chol(XF*XF');
    if cholp, return; end % show-stopper!

    [CXB,cholp] = chol(XB*XB');
    if cholp, return; end % show-stopper!

    AF(:,kf) = CXF'\I;
    AB(:,kb) = CXB'\I;

    % and loop

    while k <= p

        EF = AF(:,kf)*reshape(XX(:,1:k,k+1:m,:),kn,M); % forward  prediction errors
        EB = AB(:,kb)*reshape(XX(:,1:k,k:m-1,:),kn,M); % backward prediction errors

        [CEF,cholp] = chol(EF*EF');
        if cholp, return; end  % show-stopper!

        [CEB,cholp] = chol(EB*EB');
        if cholp, return; end  % show-stopper!

        R = CEF'\(EF*EB')/CEB; % normalised reflection coefficients

        [RF,cholp] = chol(I-R*R');
        if cholp, return; end  % show-stopper!

        [RB,cholp] = chol(I-R'*R);
        if cholp, return; end  % show-stopper!

        k  = k+1;
        kn = k*n;
        M  = N*(m-k);
        kf = 1:kn;
        kb = q1n-kn+1:q1n;

        AFPREV = AF(:,kf);
        ABPREV = AB(:,kb);

        AF(:,kf) = RF'\(AFPREV-R*ABPREV);
        AB(:,kb) = RB'\(ABPREV-R'*AFPREV);

    end

    if nargout > 1
        E   = AFPREV(:,1:n)\EF;   % residuals
        SIG = (E*E')/(M-1);       % residuals covariance matrix
        E   = reshape(E,n,m-p,N); % put residuals back into per-trial form
    end

    A = reshape(-AF(:,1:n)\AF(:,n+1:end),n,n,p); % so A(:,:,k) is the k-lag coefficients matrix

else
    error('bad regression mode ''%s''\n',regmode);
end
% Autocovariance calculation 
[G,info] = var_to_autocov(A,SIG);

function Y = demean(X,normalise)

if nargin < 2 || isempty(normalise), normalise = false; end

[n,m,N] = size(X);

U = ones(1,N*m);
Y = X(:,:);
Y = Y-mean(Y,2)*U;
if normalise
    Y = Y./(std(Y,[],2)*U);
end
Y = reshape(Y,n,m,N);

function b = isbad(x,demand_allfinite)

if nargin < 2 || isempty(demand_allfinite), demand_allfinite = true; end

if demand_allfinite
    b = ~all(isfinite(x(:)));
else
    b = ~any(isfinite(x(:)));
end

function [G,info] = var_to_autocov(A,SIG,acmaxlags,acdectol,aitr,maxiters,maxrelerr)

%global have_dlyap;

% default parameters

if nargin < 3 || isempty(acmaxlags), acmaxlags = 0;     end % calculate maximum lags automatically
if nargin < 4 || isempty(acdectol),  acdectol  = 1e-8;  end % autocovariance decay tolerance
if nargin < 5 || isempty(aitr),      aitr      = false; end % use "accelerated" iterative Lyapunov equation solver 

% iterative algorithm only: ensure defaults for utils/dlyap_aitr.m.

if nargin < 6, maxiters  = []; end
if nargin < 7, maxrelerr = []; end

[n,n1,p] = size(A);
assert(n1 == n,'VAR coefficients matrix has bad shape');
pn1 = (p-1)*n;

[nn1,nn2] = size(SIG);
assert(nn1 == nn2,'residuals covariance matrix not square');
assert(nn1 == n  ,'residuals covariance matrix doesn''t match VAR coefficients matrix');

% initialise info struct
info.error     = 0;
info.errmsg    = '';
info.warnings  = 0;
info.warnmsg   = cell(0,1);
info.rho       = NaN;
info.iters     = NaN;
info.acrelerr  = NaN;
info.acminlags = NaN;
info.aclags    = NaN;

G = [];

% construct VAR coefficients for 1-lag problem

A1 = [reshape(A,n,p*n); eye(pn1) zeros(pn1,n)];

% calculate spectral radius
try
    info.rho = max(abs(eig(A1)));
catch
    info.error = 1;
    info.errmsg = 'eig(Input matrix contains NaN or Inf.)';
    return
end

if info.rho >= 1
    info.error = 1;
    info.errmsg = 'unstable VAR (unit root)';
    return
end

% construct residual covariances for 1-lag problem

if ~isposdef(SIG);
    info.error = 2;
    info.errmsg = 'residuals covariance matrix not positive-definite';
    return;
end

SIG1 = [SIG zeros(n,pn1); zeros(pn1,n) zeros(pn1)];

% solve the Lyapunov equation for the 1-lag covariance matrix
try
    if aitr
        [G1,info.iters] = dlyap_aitr(A1,SIG1,maxiters,maxrelerr); % experimental: fast, but needs more testing
    else
%       G1 = dlyap(A1,SIG1);           % dlyap seems to work better here without balancing, which seems to break positive-definitiveness
        G1 = lyapslv('D',A1,[],-SIG1); % sometimes. However lyapslv is not an official interface, so this could conceivably break in future.
    end
catch except
    info.error = 3;
    info.errmsg = ['Lyapunov equation solver failed: ' except.message];
    return
end

info.acrelerr = norm(A1*G1*A1'-G1+SIG1)/norm(SIG1); % this should be small (see below)

maxacrelerr = 1e-8; % probably shouldn't be hard-coded :-/
if info.acrelerr > maxacrelerr
    info.warnings = info.warnings+1;
    info.warnmsg{info.warnings} = sprintf('large relative error = %g (tolerance = %g)',info.acrelerr,maxacrelerr);
end

% estimate number of autocov lags

info.acminlags = ceil(log(acdectol)/log(info.rho)); % minimum lags to achieve specified tolerance

if     acmaxlags < 0  % use exactly -acmaxlags lags (not encouraged, hence undocumented!)
    info.aclags = -acmaxlags;
elseif acmaxlags > 0  % use at most acmaxlags lags
    info.aclags = min(info.acminlags,acmaxlags);
else                  % acmaxlags == 0 - use minimum acceptable lags (recommended)
    info.aclags = info.acminlags;
end

if info.aclags < info.acminlags
    info.warnings = info.warnings+1;
    info.warnmsg{info.warnings} = sprintf('too few autocovariance lags = %d (minimum = %d)',info.aclags,info.acminlags);
end

if ~isposdef(G1);
    info.error = 4;
    info.errmsg = '1-lag covariance matrix not positive-definite';
    return
end

q = info.aclags;
q1 = q+1;

% calculate recursively from 1-lag solution (which supplies up to p-1 lags), from p lags up to q

[n,~,p]  = size(A);
assert(info.aclags >= p,'number of lags is too small'); % lags must be at least number of VAR lags
pn = p*n;
G = cat(3,reshape(G1(1:n,:),n,n,p),zeros(n,n,q1-p));   % autocov forward  sequence
B = [zeros((q1-p)*n,n); G1(:,end-n+1:end)];            % autocov backward sequence
A = reshape(A,n,pn);                                   % coefficients
for k = p:q
    r = q1-k;
    G(:,:,k+1) = A*B(r*n+1:r*n+pn,:);
    B((r-1)*n+1:r*n,:) = G(:,:,k+1);
end

function pd = isposdef(A)

[~,p] = chol(A);
pd = ~(p > 0);

function [AF,SIG] = autocov_to_var(G)

[n,~,q1] = size(G);
q = q1-1;
qn = q*n;

G0 = G(:,:,1);                                               % covariance
GF = reshape(G(:,:,2:end),n,qn)';                            % forward  autocov sequence
GB = reshape(permute(flipdim(G(:,:,2:end),3),[1 3 2]),qn,n); % backward autocov sequence

AF = zeros(n,qn); % forward  coefficients
AB = zeros(n,qn); % backward coefficients (reversed compared with Whittle's treatment)

% initialise recursion

k = 1;            % model order

r = q-k;
kf = 1:k*n;       % forward  indices
kb = r*n+1:qn;    % backward indices

AF(:,kf) = GB(kb,:)/G0;
AB(:,kb) = GF(kf,:)/G0;

% and loop

for k=2:q

    AAF = (GB((r-1)*n+1:r*n,:)-AF(:,kf)*GB(kb,:))/(G0-AB(:,kb)*GB(kb,:)); % DF/VB
    AAB = (GF((k-1)*n+1:k*n,:)-AB(:,kb)*GF(kf,:))/(G0-AF(:,kf)*GF(kf,:)); % DB/VF

    AFPREV = AF(:,kf);
    ABPREV = AB(:,kb);

    r = q-k;
    kf = 1:k*n;
    kb = r*n+1:qn;

    AF(:,kf) = [AFPREV-AAF*ABPREV AAF];
    AB(:,kb) = [AAB ABPREV-AAB*AFPREV];

end

if nargout > 1
    SIG = G0-AF*GF;
end

AF = reshape(AF,n,n,q);

function pval = mvgc_pval(x,p,m,N,nx,ny,nz,tstat)

if nargin < 7, nz    = []; end % ensure default
if nargin < 8, tstat = []; end % ensure default

pval = NaN(size(x)); % output p-value matrix is same shape as x matrix
nn   = ~isnan(x);    % indices of non-NaN x values (logical array)
x    = x(nn);        % vectorise non-NaN x values
pval(nn) = 1-mvgc_cdf(x,0,p,m,N,nx,ny,nz,tstat); % assume null hypothesis F = 0

function P = mvgc_cdf(x,X,p,m,N,nx,ny,nz,tstat)

assert(isvector(x),'evaluation values must be a vector');
n = length(x);

assert(isvector(X),'MVGC values must be a vector');
assert(all(X >= 0),'MVGC values must be non-negative');
if isscalar(X)
    X = X*ones(n,1);
else
    assert(length(X) == n,'MVGC values must match evaluation values');
end

if nargin < 8 || isempty(nz), nz = 0; end % unconditional

if nargin < 9 || isempty(tstat);
    ftest = nx == 1; % default: use F-distribution for univariate predictee, chi2 for multivariate
else
    switch lower(tstat)
        case 'f'     % Granger F-test form
            assert(nx == 1,'F-distribution is not appropriate for multivariate predictee');
            ftest = true;
        case 'chi2'  % Geweke chi2 test form
            ftest = false;
        otherwise
            error('unknown distribution (must be ''chi2'' or ''F'')');
    end
end

P = zeros(n,1);
m = N*(m-p);                  % effective number of observations (p-lag autoregression loses p observations per trial)
if ftest
    if any(X > 0), fprintf(2,'WARNING (mvgc_cdf): non-central F-distribution is experimental\n'); end
    d1 = p*ny;                % #{full model parameters} - #{reduced model parameters}
    d2 = m-p*(1+ny+nz);       % #{observations} - #{full model parameters}
    mm = d2/d1;
    for i = 1:n
        xx = exp(x(i))-1;     % Granger form: (RSS_reduced - RSS_full) / RSS_full
        if X(i) > 0           % non-central
            XX = exp(X(i))-1; % Granger form
            P(i) = ncfcdf(mm*xx,d1,d2,m*XX); % NOTE: non-centrality parameter factor might reasonably be m, d2 or d2-2
        else
            P(i) = fcdf(mm*xx,d1,d2);
        end
    end
else
    d = p*nx*ny;              % note that d does not depend on the number of conditioning variables
    for i = 1:n
        if X(i) > 0           % non-central
            P(i) = ncx2cdf(m*x(i),d,m*X(i));
        else
            P(i) = chi2cdf(m*x(i),d);
        end
    end
end

function [X,iters] = dlyap_aitr(A,Q,maxiters,maxrelerr)

if nargin < 3 || isempty(maxiters),  maxiters  = 100;  end
if nargin < 4 || isempty(maxrelerr), maxrelerr = 1e-8; end

assert(size(A,2) == size(A,1),'matrix A not square');
assert(isequal(size(Q),size(A)),'matrix Q does not match matrix A');

X  = Q;
AA = A;
snorm = norm(Q,'fro');
minrelerr = realmax;
for iters = 1:maxiters+1
    relerr = norm(X-A*X*A'-Q,'fro')/snorm;
    if relerr < maxrelerr                  % only start convergence test after max rel error threshold reached
        if relerr >= minrelerr, break; end % deemed converged
    end
    if relerr < minrelerr, minrelerr = relerr; end
    X = AA*X*AA'+X;
    AA = AA*AA;
end

if iters > maxiters
    throw(MException('MVGC:XMaxItrs','exceeded maximum iterations (max. rel. error = %e)',relerr));
end

