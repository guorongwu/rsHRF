function rsHRF_conn_job(job,data,flag_ROI, outdir, v, smask_ind, flag_nii_gii)
Nscans = size(data,1);
name = job.raw_outname;
flag_pval_pwgc = job.para_global.flag_pval_pwgc;
if ~flag_ROI % 2D/3D data
    if flag_nii_gii==1
        v0 = v(1);   v0.dt = [16,0]; v0.n = [1 1]; 
    else
        v0=v;
    end
    if ~isempty(job.connectivity)
        fprintf('Connectivity analysis...\n');
        connroinfo = rsHRF_conn_check(job);
    end
    if ~isempty(connroinfo) 
        data_nD = nan(Nscans, prod(v0.dim)); 
        data_nD(:,smask_ind) = data;
        rsHRF_conn_run(data_nD, connroinfo,v0,name,outdir,flag_pval_pwgc,flag_nii_gii);    
        clear data_nD data
    else
        warning('Please Configure Parameters for Connectivity Analysis')
    end
else %ROI-wise
    if ~isempty(job.connectivity)
        fprintf('Connectivity analysis...\n');
        connroinfo = rsHRF_conn_check(job);
        rsHRF_conn_run(data, connroinfo,[],name,outdir,1);
        clear data
    else
        warning('Please Configure Parameters for Connectivity Analysis')
    end
end

if job.job_save
    save(fullfile(outdir,[name, '_conn_job.mat']), 'job');  
end