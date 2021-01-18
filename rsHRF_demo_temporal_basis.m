%% Demo code for HRF deconvolution (Temporal Basis Functions)
clc,clear;close all;

%%===========BOLD-fMRI Data========================

load HCP_100307_rfMRI_REST1_LR_Atlas_hp2000_clean_dtseries.mat
nobs=size(bold_sig,1);  
TR = .72;
bands=[0.01 0.1]; %bandpass filter lower and upper bound
bold_sig = bold_sig-mean(bold_sig);
data = rsHRF_band_filter(bold_sig,TR,bands);
sigma = std(data);
%%===========PARAMETERS========================
para.TR = TR;
BF = {'Canonical HRF (with time derivative)'
'Canonical HRF (with time and dispersion derivatives)'              
'Gamma functions'
'Fourier set'
'Fourier set (Hanning)'};
% choose the set of basis functions THIS MUST BE AN INPUT
bf_id = 3;
para.name = BF{bf_id}; % Gamma functions
para.order = 3; % for Gamma functions or Fourier set

temporal_mask = []; % without mask, it means temporal_mask = logical(ones(nobs,1)); i.e. all time points included. nobs: number of observation = size(data,1). if want to exclude the first 1~5 time points, let temporal_mask(1:5)=0;
% temporal_mask = logical(ones(nobs,1));  temporal_mask(5:15)=0;

para.T  = 3; % magnification factor of temporal grid with respect to TR. i.e. para.T=1 for no upsampling, para.T=3 for 3x finer grid
para.T0 = 1; % position of the reference slice in bins, on the grid defined by para.T. For example, if the reference slice is the middle one, then para.T0=fix(para.T/2)
if para.T==1
    para.T0 = 1;
end

para.dt  = para.TR/para.T; % fine scale time resolution.
para.AR_lag = 1; % AR(1) noise autocorrelation.
para.thr = 1; % (mean+) para.thr*standard deviation threshold to detect event.

para.len = 24; % length of HRF, in seconds

min_onset_search = 4; % minimum delay allowed between event and HRF onset (seconds)
max_onset_search = 8; % maximum delay allowed between event and HRF onset (seconds)
para.lag  = fix(min_onset_search/para.dt):fix(max_onset_search/para.dt);

%%=============HRF estimation======================

tic

[beta_hrf, bf, event_bold] = rsHRF_estimation_temporal_basis(data,para,temporal_mask);
hrfa = bf*beta_hrf(1:size(bf,2),:); %HRF
nvar = size(hrfa,2); PARA = zeros(3,nvar);
for voxel_id=1:nvar
    hrf1 = hrfa(:,voxel_id);
    PARA(:,voxel_id) = rsHRF_get_HRF_parameters(hrf1,para.dt); % estimate HRF parameter
end

toc
%%=============HRF deconvolution======================
disp('Deconvolving HRF ...');
tic
T = round(para.len/TR);
if para.T>1
    hrfa_TR = resample(hrfa,1,para.T);
else
    hrfa_TR = hrfa;
end
hrf=hrfa_TR;

flag_deconv_raw = 1; 
if flag_deconv_raw %HRF deconvolution with raw/filtered data
    zdata = bold_sig; zscore(bold_sig);
else
    zdata =  zscore(data);
end
% data_deconv = rsHRF_iterative_wiener_deconv(zdata,hrf./sigma);
data_deconv = rsHRF_iterative_wiener_deconv(zdata,hrf,10);

event_number=length(event_bold{1,1});
toc
disp('Done');


%% example plots
event_plot=nan(1,nobs);
event_plot(event_bold{1,1})=1;
figure(1);plot((1:length(hrfa(:,1)))*TR/para.T,hrfa(:,1),'b');xlabel('Time (s)')
title(['HRF (',BF{bf_id},')'])
figure(2);plot((1:nobs)*TR,zscore(data(:,1)));
hold on;plot((1:nobs)*TR,zscore(data_deconv(:,1)),'r');
stem((1:nobs)*TR,event_plot,'k');legend('BOLD','Deconvolved BOLD','BOLD events');xlabel('Time (s)')