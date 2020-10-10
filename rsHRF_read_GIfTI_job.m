function data = rsHRF_read_GIfTI_job(job)
%read NIfTI data 
gii_file = job.images; 
vol_th = job.para_global.volume_threshold;
data = [];
fprintf('Reading GIfTI Data ...\n')
Nscans = length(gii_file);
if Nscans<vol_th
    if Nscans==1
        [pth0,nam0,ext0] = spm_fileparts(gii_file{1});
        tmp = fullfile(pth0,[nam0,ext0]);
        v  = gifti(tmp);
        data = v.cdata'; %be careful
        [nobs,nvar] = size(data);
        fprintf('(Check) input data: %d time points, #%d vertices\n',nobs,nvar) 
    else
        error(['More than one file, please check your input data: ',num2str(Nscans)]) 
    end
end
