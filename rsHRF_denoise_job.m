function [data,data_nuisancerm] = rsHRF_denoise_job(job,data)
Nscans = size(data,1);
flag_delete = job.para_global.delete_files; % delete temporary files (generated wm/csf/brainmask) 
covariates = job.Denoising.generic;
if ~isempty(covariates)
    fprintf('Reading Covariates ...\n')
    [txt,mat,nii]= wgr_check_covariates(covariates); 
    if nargin==3 % NIFTI
        if ~isempty(nii) 
            nii2 = cell(1,length(nii)); 
            for i=1:length(nii)
                nii2{1,i} = spm_file(nii{i,1},'prefix',['bin_',job.raw_outname]);
                spm_imcalc({job.ref_nii;nii{i,1}},nii2{1,i},'(i1~=0).*(i2>0.9)',{0,0,0,16});
                v0 = spm_vol(nii2{1,i});
                datam = spm_read_vols(v0);
                if flag_delete
                    delete(v0.fname)
                end
                id = find(datam==1); clear datam
                if isempty(id)
                    error(['no survived voxels (>0.9) in ',nii{i,1}])
                else
                    if nii{i,2}                                
                        numcomps = nii{i,2};
                        z = data(:,id);
                        z(isnan(z))=0;
                        m = size(z,2) ;
                        if m<numcomps
                            error(['# components = ',num2str(m),', < ', num2str(numcomps)])
                        else
                            pc = pca(z);
                        end
                        nii2{1,i} = z*pc(:,1:numcomps);
                    else
                        nii2{1,i} = nanmean(data(:,id),2);
                    end
                end
            end
            nii = cell2mat(nii2);
            clear nii2 datam
        else
            nii = [];
        end
        nuisance = [txt mat nii];
    elseif nargin==2
        nuisance = [txt mat];
    end
else
    nuisance = [];
end

%% Denoising parameters
p_detrend = job.Denoising.Detrend; 
if p_detrend
    nuisance_detrend = zeros(Nscans,p_detrend);
    for i = 1:p_detrend
        d = (1:Nscans).^i;
        nuisance_detrend(:,i+1) = d(:)./(Nscans.^i);
    end
else
    nuisance_detrend= [];
end

if ~isempty(nuisance) && ~isempty(nuisance_detrend)
    fprintf('Removing nuisance regressors & Detrend................. \n ')
elseif isempty(nuisance) && ~isempty(nuisance_detrend)
    fprintf('Detrend................. \n ')
elseif  ~isempty(nuisance) &&  isempty(nuisance_detrend)
    fprintf('Removing nuisance regressors................. \n ')
end
nuisance = [nuisance_detrend nuisance];
if ~isempty(nuisance)
    nuisance = bsxfun(@minus, nuisance, sum(nuisance, 1)./Nscans);% Remove the mean
    nuisance = [orth(nuisance) ones(Nscans, 1)];
    mu   = sum(data, 1)./Nscans;    % mean 
    data = data - nuisance*(nuisance\data);
    data = bsxfun(@plus, data, mu);
end
data_nuisancerm = data;

BPF = job.Denoising.BPF;
[Bands,TR] = wgr_check_bpf(BPF);
if ~isempty(Bands)
    fprintf('Frequency filter.................... \n ')
    if isnan(TR)
        if isfield(job,'HRFE')
            TR = job.HRFE.TR ; 
        else
            error('Please add TR information for Filtering')
        end
    else
        if isfield(job,'HRFE')
            if TR ~= job.HRFE.TR ; 
                error('Different TR for HRF estimation and Filtering')
            end
        end
    end       
    data = rsHRF_band_filter(data,TR,Bands);
end

if job.Denoising.Despiking
    fprintf(' Temporal despiking with a hyperbolic tangent squashing function... \n')
    data = wgr_despiking(data);
end


function x1 = wgr_despiking(x1)
my=repmat(median(x1,1),[size(x1,1),1]);
sy=repmat(4*median(abs(x1-my)),[size(x1,1),1]);
x1=my+sy.*tanh((x1-my)./max(eps,sy));

function [Bands,TR] = wgr_check_bpf(BPF)
if length(BPF)==1
    try
        Bands = BPF{1}.bands;
        TR = nan;
    catch
        error('please check your setting for band-pass filtering \n')
    end
elseif length(BPF)==2
    try
        Bands = BPF{1}.bands;
        TR = BPF{2}.TR;
    catch
        try
            Bands = BPF{2}.bands;
            TR = BPF{1}.TR;
        catch
            error('please check your setting for band-pass filtering \n')
        end
    end
else
    Bands =[]; TR=nan;
end

function [txt,mat,nii]= wgr_check_covariates(covariates)
txt={};  mat = {};  nii  = {};
k1 =1 ; k2 = 1; k3 = 1;
for i=1:length(covariates)
    tmp  = covariates{i};
    flag = isfield(tmp,'multi_reg');
    if flag
        for j=1: length(tmp.multi_reg)
            tmp2 = strcat(tmp.multi_reg{j});
            [~,~,ext] = fileparts(tmp2);
            if ~isempty(strfind(ext,'txt'))
                txt{1,k1} = load(tmp2);
                k1 = k1+1;
            elseif ~isempty(strfind(ext,'mat'))                
                tmp3  = load(tmp2);
                try
                    mat{1,k2} = tmp3.R;
                catch
                    tmp4 = fieldnames(tmp3);
                    if length(tmp4)==1
                        eval(['mat{1,k2} = tmp3.',tmp4{1},';']);
                    else
                        error('there are more than one variable in your mat file');
                    end
                end
                k2 = k2+1;
            else
                error('Unknown File Types')
            end
            continue;
        end
    end
        
    flag = isfield(tmp,'imcov');
    if flag
        for j=1: length(tmp.imcov.images)
            tmp2 = strcat(tmp.imcov.images{j});
            nii{k3,1} = tmp2;
            if length(tmp.imcov.mpc)==length(tmp.imcov.images)
                nii{k3,2} = tmp.imcov.mpc(j);
            elseif length(tmp.imcov.mpc)==1
                nii{k3,2} = tmp.imcov.mpc(1);
            end
            k3 = k3+1;
        end
        continue;
    end
end
txt = cell2mat(txt);
mat = cell2mat(mat);