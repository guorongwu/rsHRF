function rsHRF_ROI_sig_job(job)
ROI = job.Datasig; %cell file
[data_txt,mat_name]= rsHRF_check_ROIsig(ROI);
tmp = data_txt(:,1);
if job.para_global.combine_ROI
    data = cell2mat(tmp'); % combine all ROI together 
    tmp={}; tmp{1} = data;
    [~,outname,~] = fileparts(data_txt{1,2});
    fprintf('Combine all input ROIs for connectivity analysis, output is "*comb_%s*.mat"\n',outname)
end
nROI = length(tmp);
for i=1:nROI
    data = tmp{i};
    if isempty(data_txt{1,2})
        error('No data input!')
    else
        [fpath,name,~] = fileparts(data_txt{1,2});
        if job.para_global.combine_ROI
            job.raw_outname = ['combROI_',name];
        else
            if isempty(mat_name{i})
                job.raw_outname = name;
            else
                job.raw_outname = [name,'_',mat_name{i}];
            end
        end
    end
    outdir = job.outdir{1};
    if isempty(outdir)
        outdir = fpath;
    else
        if ~exist(outdir,'dir')
            mkdir(outdir)
        end
    end
    [data,data_nuisancerm] = rsHRF_denoise_job(job,data);
    flag_ROI = 1;
    
    if isfield(job,'HRFE') % deconvolution
        rsHRF_deconv_job(job,data,data_nuisancerm,flag_ROI, outdir);
    else
        rsHRF_conn_job(job,data,flag_ROI, outdir);
    end
end


function [data_txt,var_name]= rsHRF_check_ROIsig(ROI)
nROI = length(ROI);
data_txt = cell(nROI,2);
var_name  = cell(nROI,1);
for i=1:nROI
    tmp2 = ROI(i).sigdata{1};
    [~,~,ext] = fileparts(tmp2);
    if strfind(ext,'txt')
        data_txt{i,1} = load(tmp2);
        data_txt{i,2} = tmp2;
    elseif strfind(ext,'mat')
        tmp3 = load(tmp2);
        var_name{i} = strcat(ROI(i).name);
        try
            eval(['data_txt{i,1} = tmp3.',var_name{i},';']);
        catch
            tmp4 = fieldnames(tmp3);
            if length(tmp4)==1
                eval(['data_txt{i,1} = tmp3.',tmp4{1},';']);
            else
                error(['there are more than one variable in mat file, and no variable ''',var_name,'" in mat file']);
            end
        end
        data_txt{i,2} = tmp2;
    end
end