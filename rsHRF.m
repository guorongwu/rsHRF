function rsHRF(varargin)
%% resting state BOLD-fMRI HRF deconvolution and connectivity analysis
if nargin==2
    job = varargin{1};  flag = varargin{2};  
    if ~isfield(job,'para_global')
        job.para_global =  rsHRF_global_para;
    end
    switch flag
        case 'display'
            rsHRF_viewer(job);
        case 'volume'
            % delete temporary files (generated wm/csf/brainmask) 
            flag_delete = job.para_global.delete_files; 
            
            %input images, cell file
            nii_file = job.images; 
            
            %get 1st file name (out name)
            ref_nii = nii_file{1}; %funtional space template
            [fpath,raw_outname,~] = fileparts(ref_nii);
            job.ref_nii = ref_nii;
            job.raw_outname = raw_outname;
            
            %out directory
            outdir = job.outdir{1};
            if isempty(outdir)
                outdir = fpath;
            else
                if ~exist(outdir,'dir')
                    mkdir(outdir)
                end
            end
            
            % NIfTI data 
            [data,v] = rsHRF_read_NIfTI_job(job);

            % explicit mask
            nii_mask = job.mask{1};
            if isempty(nii_mask)
                nii_mask = job.para_global.mask_nii ;
            end      
            if ~isempty(nii_mask)  %brain mask
                nii_masknew  = spm_file(nii_mask,'prefix',['bin_',raw_outname]);
                spm_imcalc({ref_nii;nii_mask},nii_masknew,'(i1~=0).*(i2~=0)',{0,0,0,16});
                v0 = spm_vol(nii_masknew);
                datam = spm_read_vols(v0);
                datam(isnan(datam))=0;
                voxel_ind = find(datam~=0); clear datam
                if flag_delete
                    delete(v0.fname)
                end
            else % var>0 as mask
                data_var = nanvar(data,0,1);
                voxel_ind = find(data_var>0); clear data_var
            end
            fprintf('# %d voxels inside defaut mask ...\n',length(voxel_ind))

            
            % check ROI information
            if  isfield(job,'genericROI')    
                ROI = job.genericROI;
            else
                ROI = [];
            end
            
            if ~isempty(ROI)
                flag_ROI = 1;
                [mat,atlas,nii]= rsHRF_check_ROI(ROI, ref_nii,flag_delete);                
                roinfo = [mat;atlas;nii];
                nvar = size(roinfo,1);
                Nscans = size(data,1);
                if job.Denoising.which1st == 3;
                    job.Denoising.which1st = 2;
                end
                if job.Denoising.which1st==2
                    data_ROI = zeros(Nscans,nvar);
                    for j=1:nvar
                        data_ROI(:,j) = nanmean(data(:,roinfo{j,1}),2);
                    end                
                    [data,data_nuisancerm] = rsHRF_denoise_job(job,data_ROI);
                elseif job.Denoising.which1st==1
                    [data,data_nuisancerm] = rsHRF_denoise_job(job,data);
                    data_ROI = zeros(Nscans,nvar);
                    data_ROI_nuisancerm = zeros(Nscans,nvar);
                    for j=1:nvar
                        data_ROI(:,j) = nanmean(data(:,roinfo{j,1}),2);
                        data_ROI_nuisancerm(:,j) = nanmean(data_nuisancerm(:,roinfo{j,1}),2);
                    end
                    data = data_ROI;
                    data_nuisancerm = data_ROI_nuisancerm;
                    clear data_ROI data_ROI_nuisancerm                    
                end
            else
                flag_ROI=0;
                data = data(:,voxel_ind);
                [data,data_nuisancerm] = rsHRF_denoise_job(job,data);
            end

            if flag_ROI
                if isfield(job,'prefix')
                    if job.savedata.job_save
                        save(fullfile(outdir,[job.prefix,raw_outname, '.mat']), 'roinfo','-v7.3');
                    end
                else
                    if job.job_save
                        save(fullfile(outdir,[raw_outname, '_roinfo.mat']), 'roinfo');
                    end
                end                
            end
           
            flag_nii_gii = 1; %NIfTI
            if isfield(job,'HRFE') % deconvolution
                rsHRF_deconv_job(job,data,data_nuisancerm,flag_ROI, outdir, v, voxel_ind, flag_nii_gii);
            else
                rsHRF_conn_job(job,data,flag_ROI, outdir, v, voxel_ind, flag_nii_gii)
            end
            
        case 'mesh'    
            %input images, cell file
            gii_file = job.images; 
            [fpath,raw_outname,~] = fileparts(gii_file{1});
            job.raw_outname = raw_outname;
            job.rmoutlier = 0;
            %out directory
            outdir = job.outdir{1};
            if isempty(outdir)
                outdir = fpath;
            else
                if ~exist(outdir,'dir')
                    mkdir(outdir)
                end
            end
            % GIfTI data 
            data = rsHRF_read_GIfTI_job(job);
            v.dim = [size(data,2), 1];
            % explicit mask
            gii_mask = job.mask{1};

            if ~isempty(gii_mask)  %brain mask
                datam = gifti(gii_mask);
                datam.cdata(isnan(datam.cdata))=0;
                vertex_ind = find(datam.cdata~=0); clear datam
            else % var>0 as mask
                data_var = nanvar(data,0,1);
                vertex_ind = find(data_var>0); clear data_var
            end
            fprintf('# %d vertexs inside the mask ...\n',length(vertex_ind))

            % check ROI information
            if  isfield(job,'genericROI')    
                ROI = job.genericROI;
            else
                ROI = [];
            end

            if ~isempty(ROI)
                flag_ROI = 1;
                [~,~,~,atlasmeh,gii]= rsHRF_check_ROI(ROI);                
                roinfo = [atlasmeh; gii];
                nvar = size(roinfo,1);
                Nscans = size(data,1);
                if job.Denoising.which1st == 3;
                    job.Denoising.which1st = 2;
                end
                if job.Denoising.which1st==2
                    data_ROI = zeros(Nscans,nvar);
                    for j=1:nvar
                        data_ROI(:,j) = nanmean(data(:,roinfo{j,1}),2);
                    end
                    [data,data_nuisancerm] = rsHRF_denoise_job(job,data_ROI);

                elseif job.Denoising.which1st==1

                    [data,data_nuisancerm] = rsHRF_denoise_job(job,data);
                    data_ROI = zeros(Nscans,nvar);
                    data_ROI_nuisancerm = zeros(Nscans,nvar);
                    for j=1:nvar
                        data_ROI(:,j) = nanmean(data(:,roinfo{j,1}),2);
                        data_ROI_nuisancerm(:,j) = nanmean(data_nuisancerm(:,roinfo{j,1}),2);
                    end
                    data  = data_ROI;
                    data_nuisancerm  = data_ROI_nuisancerm;
                    clear data_ROI data_ROI_nuisancerm
                end
            else
                flag_ROI=0;
                data = data(:,vertex_ind);
                [data,data_nuisancerm] = rsHRF_denoise_job(job,data);
            end
            
            if flag_ROI
                if isfield(job,'prefix')
                    if job.savedata.job_save
                        save(fullfile(outdir,[job.prefix,raw_outname, '.mat']), 'roinfo','-v7.3');
                    end
                else
                    if job.job_save
                        save(fullfile(outdir,[raw_outname, '_roinfo.mat']), 'roinfo');
                    end
                end
            end
                        
            flag_nii_gii = 2; %GIfTI
            if isfield(job,'HRFE') % deconvolution
                rsHRF_deconv_job(job,data,data_nuisancerm,flag_ROI, outdir, v, vertex_ind, flag_nii_gii);
            else
                rsHRF_conn_job(job,data,flag_ROI, outdir, v, vertex_ind, flag_nii_gii)
            end
        case 'sig'
            rsHRF_ROI_sig_job(job)         
    end
    return
else
    code_path = fileparts(which('rsHRF.m')) ;
    a = imread(fullfile(code_path,'rsHRF_logo.png'));
    pos = [0.4   0.3    0.2    0.3236];
    if ismac       % Code to run on Mac plaform
    elseif isunix  % Code to run on Linux plaform
        pos = [0.4   0.3    0.23    0.35];
    elseif ispc    % Code to run on Windows platform
    end

    S.fig = figure('Visible','on',...
        'numbertitle','off',...    
        'menubar','none',...         
        'units','normalized',...
        'color','w',...
        'position',pos,...[563    98   480   360],... %0.4   0.3    0.15    0.38
        'name',['rsHRF v2.3(',getenv('USERNAME'),')'],...
        'resize','off');
    
    axes('parent',S.fig,'units','normalized','position',[0.15 0.45 [1 0.7825]*0.76]); %[0 0.59 1 0.43]
    imagesc(a, 'AlphaData', 0.6);
    axis off; 
    
    %% ROI signal HRF deconvolution
    S.pb(1) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.05,0.05,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','Signals',...
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.sig_rsHRF'');',...
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');

    %% ROI-wise HRF deconvolution
    S.pb(2) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.05,0.2,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','ROIs-volume',...
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.ROI_rsHRF'');',...        
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');
    
    %% (Surface) ROI-wise HRF deconvolution
    S.pb(3) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.55,0.2,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','ROIs-surface',...
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.SurfROI_rsHRF'');',...        
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');
    
    %% voxelwise HRF deconvolution
    S.pb(4) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.05,0.35,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','Voxels',...
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.vox_rsHRF'');',...        
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');
    %% vertexwise HRF deconvolution
    S.pb(5) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.55,0.35,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','Vertices',...
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.mesh_rsHRF'');',...        
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');
    
    S.pb(6) = uicontrol('parent',S.fig,...
        'unit','norm',...
        'pos',[0.55,0.05,0.4,0.12],...
        'style','pushbutton',...'pushbutton',...
        'string','Display',...        
        'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.display_HRF'');',...        
        'backgroundc','w',...
        'foregroundcolor',0*[1 1 1],...     'fontname','Segoe Script',...
        'fontname','Calibri',...'Times New Roman',...
        'fontunits', 'normalized',... 
        'fontsize', 0.4,... 
        'fontweight','bold');
    
    if nargin==1
        Modality=upper(varargin{1});
        switch Modality
            case 'CONN'  
                set(S.pb(1),'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.sig_conn'');'); 
                set(S.pb(2),'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.ROI_conn'');');
                set(S.pb(3),'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.SurfROI_conn'');');
                set(S.pb(4),'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.vox_conn'');');
                set(S.pb(5),'CallBack','spm_jobman(''interactive'','''',''spm.tools.rsHRF.mesh_conn'');');
                set(S.pb(6),'enable','off')
                set(S.pb(:),'foregroundcolor',0*[1 1 1]) %0.3 0.6 0
                text(80,360,'Connectivity Analysis','Color',[0.3 0.6 0],'Fontsize',13,'Fontweight','bold');
            otherwise
                text(80,360,'Resting State HRF','Color',[1 0.6 0],'Fontsize',13,'Fontweight','bold');
        end
    else
        text(80,360,'Resting State HRF','Color',[1 0.6 0],'Fontsize',13,'Fontweight','bold');
    end
                                                                                                                                                      
end

