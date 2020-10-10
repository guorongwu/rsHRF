function spm_rsHRF
clc
rev = '$Rev: 01 $';

spm('FnBanner',mfilename,rev);
[~,Fgraph,~] = spm('FnUIsetup','rsHRF toolbox');
spm_help('!Disp','rsHRF.man','',Fgraph,'Resting state BOLD fMRI HRF deconvolution toolbox for SPM');
rsHRF
