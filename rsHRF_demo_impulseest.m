%% Demo code for HRF deconvolution (Nonparametric impulse response estimations)
clc,clear;close all;

%%===========BOLD-fMRI Data========================

load HCP_100307_rfMRI_REST1_LR_Atlas_hp2000_clean_dtseries.mat
bold_sig = double(bold_sig); % double
nobs=size(bold_sig,1);  
TR = .72;
bands=[0.01 0.1]; %bandpass filter lower and upper bound
data = rsHRF_band_filter(bold_sig,TR,bands);

%%===========PARAMETERS========================
para.TR = TR;
options = impulseestOptions; % see impulseestOptions.m for help 
options.RegularizationKernel = 'none'; %Regularizing kernel, used for regularized estimates of impulse response for all input-output channels. Regularization reduces variance of estimated model coefficients and produces a smoother response by trading variance for bias
para.options = options;

temporal_mask = []; % without mask, it means temporal_mask = logical(ones(nobs,1)); i.e. all time points included. nobs: number of observation = size(data,1). if want to exclude the first 1~5 time points, let temporal_mask(1:5)=0;
% temporal_mask = logical(ones(nobs,1));  temporal_mask(5:15)=0;

para.T  = 1; 
para.T0 = 1; 
if para.T>1| para.T0>1
    para.T = 1; para.T0 = 1;
end

para.dt  = para.TR/para.T; % fine scale time resolution.
para.thr = 1; % (mean+) para.thr*standard deviation threshold to detect event.

para.len = 24; % length of HRF, in seconds

min_onset_search = 5; % minimum delay allowed between event and HRF onset (seconds)
max_onset_search = 8; % maximum delay allowed between event and HRF onset (seconds)
para.lag  = fix(min_onset_search/para.dt):fix(max_onset_search/para.dt);

%%=============HRF estimation======================

tic
[data,mu,sigma]=zscore(data);
[beta_hrf, event_bold] = rsHRF_estimation_impulseest(data,para);

hrfa = beta_hrf; %HRF
nvar = size(hrfa,2); PARA = zeros(3,nvar);
for voxel_id=1:nvar
    hrf1 = hrfa(:,voxel_id);
    PARA(:,voxel_id) = rsHRF_get_HRF_parameters(hrf1,para.dt); % estimate HRF parameter
end

toc

%%=============HRF deconvolution======================
disp('Deconvolving HRF ...');
tic
hrf=hrfa(:,1);

flag_deconv_raw = 1; 
if flag_deconv_raw %HRF deconvolution with raw/filtered data
    zdata = zscore(bold_sig);
else
    zdata =  data;
end
data_deconv = rsHRF_iterative_wiener_deconv(zdata,hrf);

event_number=length(event_bold{1,1});
toc
disp('Done');


%% example plots
event_plot=nan(1,nobs);
event_plot(event_bold{1,1})=1;
figure(1);plot((1:length(hrfa(:,1)))*TR/para.T,hrfa(:,1),'b');xlabel('Time (s)')
title('HRF (nonparametric impulse response estimation)')
figure(2);plot((1:nobs)*TR,zscore(data(:,1)));
hold on;plot((1:nobs)*TR,zscore(data_deconv(:,1)),'r');
stem((1:nobs)*TR,event_plot,'k');legend('BOLD','Deconvolved BOLD','BOLD events');xlabel('Time (s)')