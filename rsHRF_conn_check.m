function [connroinfo,conndata_flag] = rsHRF_conn_check(job)
numBC = length(job.connectivity);
connroinfo = cell(numBC,5);
conndata_flag = zeros(numBC,1);
for j=1:numBC
    if isfield(job.connectivity{j},'conn')
        conns = job.connectivity{j}.conn;
    elseif isfield(job.connectivity{j},'connG')
        conns = job.connectivity{j}.connG;
    elseif isfield(job.connectivity{j},'connP')
        conns = job.connectivity{j}.connP;
    end
    if isfield(conns,'data4conn')
        conndata_flag(j,1) =conns.data4conn;
    end
    if isfield(conns,'genericROI')
        genericROI =conns.genericROI;
        if ~isfield(job,'ref_nii');
            job.ref_nii = job.images{1};
        end
        [mat,atlas,nii,atlasmeh,gii]= rsHRF_check_ROI(genericROI, job.ref_nii,job.para_global.delete_files);  
        connroinfo{j,1} = [mat;atlas;nii;atlasmeh;gii];
        connroinfo{j,2} =conns.Seed_ROI;
    else
        connroinfo{j,1} = [];
        connroinfo{j,2} = 1;
    end
    connroinfo{j,3} =conns.method;
    if isfield(conns,'morder')
        connroinfo{j,4} =conns.morder;
    end
    if isfield(conns,'ndinfo')
        connroinfo{j,5} =conns.ndinfo;
    end
    connroinfo{j,6} =conns.prefix;
end
