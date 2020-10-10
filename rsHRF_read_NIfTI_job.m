function [data,v] = rsHRF_read_NIfTI_job(job)
%read NIfTI data 
nii_file = job.images; 
vol_th = job.para_global.volume_threshold;
data = [];
fprintf('Reading NIfTI Data ...\n')
Nscans = length(nii_file);
if Nscans<vol_th
    if Nscans==1
        [pth0,nam0,ext0] = spm_fileparts(nii_file{1});
        tmp = fullfile(pth0,[nam0,ext0]);
        v  = spm_vol(tmp);
        Nscans = length(v);
        if Nscans>vol_th
            for i=1:Nscans
                nii_file{i,1}  = [v(1).fname,',',num2str(i)];
            end
        end
    else
        error(['Please check your input data: ',num2str(Nscans),' volumes ? (set frames "inf" for 4D NIfTI data)'])
    end
end
for i=Nscans:-1:1
    v(i) = spm_vol(nii_file{i});
    tmp  = spm_read_vols(v(i));
    data(i,:) = tmp(:);
end       