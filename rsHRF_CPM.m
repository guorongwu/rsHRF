function [y_predict, performance,pmask_kfolds] = rsHRF_CPM(x,y,kfolds,pthresh,confound,pm)
% Performs Predictive Modeling 
% REQUIRED INPUTS
%        x            Predictor variable (nsubs x nvar) 
%        y            Outcome variable (e.g., behavioral scores) (nsubs x i)
%        'kfolds'     Number of partitions for dividing the sample
%                    (e.g., 2 =split half, 10 = ten fold)
% OUTPUTS
%        y_predict    Predictions of outcome variable
%        performance  Correlation between predicted and actual values of y
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2020.06.03. modified from https://github.com/YaleMRRC/CPM
% https://www.nitrc.org/projects/bioimagesuite/
if nargin<5
    confound=[];
    pm=[];
elseif nargin<6
    pm=[];
end
% Split data
nsubs=size(x,1);
randinds=randperm(nsubs); %random index
y_predict = zeros(nsubs, 1);
pmask_kfolds=[];
for leftout = kfolds:-1:1
    if kfolds == nsubs % doing leave-one-out
        testinds=randinds(leftout);
        traininds=setdiff(randinds,testinds);
    else
        testinds=randinds(leftout:kfolds:nsubs);
        traininds=setdiff(randinds,testinds);
    end
    
    %% Assign x and y data to train and test groups 
    x_train = x(traininds,:);
    y_train = y(traininds,:);
    x_test = x(testinds,:);
    confound_train = confound(traininds,:);
    %% Train Connectome-based Predictive Model
    [~, ~, pmask, mdlp,mdln] = rsHRF_cpm_train_pn(x_train, y_train, confound_train, pthresh,pm);
    
    pmaskp = +(pmask>0);
    pmaskn = +(pmask<0);
    pmask_kfolds(leftout,:) = pmask;
    %% Test Connectome-based Predictive Model    
    if any(pmaskp) || any(mdlp)
        [y_predict(testinds,1)] = rsHRF_cpm_test(x_test,mdlp,pmaskp);
    else
        [y_predict(testinds,1)] = 0;
    end
    if any(pmaskn) || any(mdln)
        [y_predict(testinds,2)] = rsHRF_cpm_test(x_test,mdln,pmaskn);
    else
        [y_predict(testinds,2)] = 0;
    end
    
end

%% Assess performance
[performance(:,1),performance(:,2)]=corr(y_predict,y);
performance(isnan(performance(:,1)),1) = -1;
performance(isnan(performance(:,2)),2) = 1;
fprintf('#')


function [r,p,pmask,mdlp,mdln]=rsHRF_cpm_train_pn(x,y,confound,pthresh,pm)
% Train a Connectome-based Predictive Model
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% r            Correlations between all x and y
% p            p-value of correlations between x and y
% pmask        Mask for significant features
% mdl          Coefficient fits for linear model relating summary features to y

% Select significant features
if isempty(confound)
    [r,p]=corr(x,y);
else
    [r,p]=partialcorr(x,y,confound);
end
pmask=(+(r>0))-(+(r<0)); 
if isempty(pm)
    pmask=pmask.*(+(p<pthresh));
else
    pmask=pmask.*(+(p<pthresh)).*pm(:);
end

% For each subject, summarize selected features
summary_feature=[];
for i=1:size(x,1)
    pv = nanmean(x(i,pmask>0)); %positive
    nv = nanmean(x(i,pmask<0)); %negative
    pv(isnan(pv)) = 0;
    nv(isnan(nv)) = 0;
    summary_feature(i,1)= pv;
    summary_feature(i,2)= nv; 
end

% Fit y to summary features
num = length(y);
if any(summary_feature(:,1))
    try
        mdlp=robustfit(summary_feature(:,1),y');    
    catch
        mdlp = regress(y, [ones(num, 1), summary_feature(:,1)]);
    end
else
    mdlp = [regress(y, ones(num, 1)); 0];
end

if all(summary_feature(:,2))
    try
        mdln=robustfit(summary_feature(:,2),y');   
    catch
        mdln = regress(y, [ones(num, 1), summary_feature(:,2)]);
    end
else
    mdln = [regress(y, ones(num, 1)); 0];
end


function [y_predict]=rsHRF_cpm_test(x,mdl,pmask)
% Test a Connectome-based Predictive Model using previously trained model
% x            Predictor variable
% mdl          Coefficient fits for linear model relating summary features to y
% pmask        Mask for significant features
% y_predict    Predicted y values

% For each subject, create summary feature and use model to predict y
for i=size(x,1):-1:1
    sf = nanmean(x(i,pmask>0));
    sf(isnan(sf)) = 0;
    y_predict(i)=mdl(2)*sf + mdl(1);
end