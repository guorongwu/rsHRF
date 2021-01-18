%% Demo code for HRF deconvolution ( (Smooth) Finite Impulse Response estimation)
clc,clear;close all;

%%===========BOLD-fMRI Data========================

load HCP_100307_rfMRI_REST1_LR_Atlas_hp2000_clean_dtseries.mat
nobs=size(bold_sig,1);  
TR = .72;
bands=[0.01 0.1]; %bandpass filter lower and upper bound
data = rsHRF_band_filter(bold_sig,TR,bands);
sigma = std(data);

%%===========PARAMETERS========================
para.TR = TR;
BF = {'FIR'
'sFIR'};
bf_id = 1;
para.estimation = BF{bf_id}; % sFIR


temporal_mask = []; % without mask, it means temporal_mask = logical(ones(nobs,1)); i.e. all time points included. nobs: number of observation = size(data,1). if want to exclude the first 1~5 time points, let temporal_mask(1:5)=0;
% temporal_mask = logical(ones(nobs,1));  temporal_mask(5:15)=0;

para.T  = 1; 
para.T0 = 1; 
if para.T>1| para.T0>1
    para.T = 1; para.T0 = 1;
end

para.dt  = para.TR/para.T; % fine scale time resolution.
para.AR_lag = 1; % AR(1) noise autocorrelation.
para.thr = 1; % (mean+) para.thr*standard deviation threshold to detect event.

para.len = 20; % length of HRF, in seconds

min_onset_search = 4; % minimum delay allowed between event and HRF onset (seconds)
max_onset_search = 8; % maximum delay allowed between event and HRF onset (seconds)
para.lag  = fix(min_onset_search/para.dt):fix(max_onset_search/para.dt);

%%=============HRF estimation======================

tic

[beta_hrf, event_bold] = rsHRF_estimation_FIR(data,para,temporal_mask);
hrfa = beta_hrf(1:end-2,:); %HRF
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
    zdata =  zscore(data);
end
data_deconv = rsHRF_iterative_wiener_deconv(zdata,hrf./sigma);

event_number=length(event_bold{1,1});
toc
disp('Done');


%% example plots
event_plot=nan(1,nobs);
event_plot(event_bold{1,1})=1;
figure(1);plot((1:length(hrfa(:,1)))*TR,hrfa(:,1),'b');xlabel('Time (s)')
title(['HRF (',BF{bf_id},')'])
figure(2);plot((1:nobs)*TR,zscore(data(:,1)));
hold on;plot((1:nobs)*TR,zscore(data_deconv(:,1)),'r');
stem((1:nobs)*TR,event_plot,'k');legend('BOLD','Deconvolved BOLD','BOLD events');xlabel('Time (s)')