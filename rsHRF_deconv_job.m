function out = rsHRF_deconv_job(job,data,data_nuisancerm,flag_ROI, outdir, v, smask_ind, flag_nii_gii)
[Nscans,nvar_data] = size(data);
name = job.raw_outname;
flag_pval_pwgc = job.para_global.flag_pval_pwgc;
flag_save_psc = job.para_global.flag_save_psc;
num_iterations = job.para_global.num_iterations;
flag_parfor = job.para_global.flag_parfor;
%% HRF deconvolution
para.TR = job.HRFE.TR ; 
%%% the following parameter (upsample grid) can be > 1 only for Canonical/Fourier/Gamma.
%%% Set = 1 for FIR/sFIR
para.T  = job.HRFE.fmri_t; % magnification factor of temporal grid with respect to TR. i.e. para.T=1 for no upsampling, para.T=3 for 3x finer grid
para.T0 = job.HRFE.fmri_t; % position of the reference slice in bins, on the grid defined by para.T. For example, if the reference slice is the middle one, then para.T0=fix(para.T/2)
if para.T==1
    para.T0 = 1;
end
min_onset_search = min(job.HRFE.mdelay); % minimum delay allowed between event and HRF onset (seconds)
max_onset_search = max(job.HRFE.mdelay); % maximum delay allowed between event and HRF onset (seconds)
para.dt  = para.TR/para.T; % fine scale time resolution.
para.order = job.HRFE.num_basis;
para.AR_lag = job.HRFE.cvi; % AR(1) noise autocorrelation.
para.thr = job.HRFE.thr; % (mean+) para.thr*standard deviation threshold to detect event.
para.len = job.HRFE.hrflen; % length of HRF, in seconds
para.lag  = fix(min_onset_search/para.dt):fix(max_onset_search/para.dt);
para.localK = job.HRFE.localK;
if isnan(job.HRFE.tmask(1))
    temporal_mask = true(Nscans,1); % temporal mask.
else
    temporal_mask = logical(job.HRFE.tmask);
end

if job.HRFE.hrfm<6
    hrfmn = {
    'Canonical HRF (with time derivative)'
    'Canonical HRF (with time and dispersion derivatives)'              
    'Gamma functions'
    'Fourier set'
    'Fourier set (Hanning)'}; % 
    para.name = hrfmn{job.HRFE.hrfm};
    tic
    [beta_hrf, bf, event_bold] = rsHRF_estimation_temporal_basis(data,para, temporal_mask,flag_parfor);
    hrfa = bf*beta_hrf(1:size(bf,2),:); %HRF
    hrf_baseline = beta_hrf(1+size(bf,2),:); %HRF baseline value for PSC calculation.
else 
    tic
    if  job.HRFE.hrfm==6
        para.estimation = 'FIR';
    elseif job.HRFE.hrfm==7
        para.estimation = 'sFIR';
    end
    para.T=1; % this needs to be = 1 for FIR
    [beta_hrf,event_bold] = rsHRF_estimation_FIR(data,para, temporal_mask,flag_parfor);
    hrfa = beta_hrf(1:end-2,:); %HRF
    hrf_baseline = beta_hrf(end-1,:); %HRF baseline value for PSC calculation.
end
 
nvar = size(hrfa,2);
PARA = zeros(3,nvar);
event_number = nan(nvar,1);
if flag_parfor
    parfor ii=1:nvar
        hrf1 = hrfa(:,ii);
        PARA(:,ii) = rsHRF_get_HRF_parameters(hrf1,para.TR/para.T);% estimate HRF parameter
        event_number(ii)=length(event_bold{1,ii});
    end
else
    for ii=1:nvar
        hrf1 = hrfa(:,ii);
        PARA(:,ii) = rsHRF_get_HRF_parameters(hrf1,para.TR/para.T);% estimate HRF parameter
        event_number(ii)=length(event_bold{1,ii});
    end
end
fprintf('\nDone HRF estimation %8.2f seconds\n',toc)
 
if job.savedata.hrfmat_save
    fprintf('Save HRF parameters....\n');
    %% save HRF parameters
    save(fullfile(outdir,[job.prefix,name, '_hrf.mat']), 'para', 'beta_hrf', 'hrfa', 'event_bold', 'event_number','PARA','-v7.3');
    if exist('bf','var')
        save(fullfile(outdir,[job.prefix,name, '_hrf.mat']), 'bf','-append')
    end
end
 
 if ~isempty(job.connectivity)
     [connroinfo,conndata_flag] = rsHRF_conn_check(job);
 else
     connroinfo={}; conndata_flag=0;
 end
    
%% HRF deconvolution -->signals
%whether save the deconvolved signal or perform analysis on deconvolved data
if job.savedata.deconv_save  || any(conndata_flag~=1) 
    fprintf('Deconvolving HRF ...\n');
    tic
%     T = round(para.len/TR);
    if para.T>1
        hrfa_TR = resample(hrfa,1,para.T);
    else
        hrfa_TR = hrfa;
    end
    data_deconv = nan(size(data));
    sigma = std(data);
    if job.HRFE.hrfdeconv==1
        if flag_parfor
            parfor ii=1:nvar
                hrf=hrfa_TR(:,ii);
                data_deconv(:,ii) = rsHRF_iterative_wiener_deconv(zscore(data_nuisancerm(:,ii)),hrf./sigma(ii),num_iterations);
            end
        else
            for ii=1:nvar
                hrf=hrfa_TR(:,ii);
                data_deconv(:,ii) = rsHRF_iterative_wiener_deconv(zscore(data_nuisancerm(:,ii)),hrf./sigma(ii),num_iterations);
            end
        end
    elseif job.HRFE.hrfdeconv==2
        if flag_parfor
            parfor ii=1:nvar
                hrf=hrfa_TR(:,ii);
                data_deconv(:,ii) = rsHRF_iterative_wiener_deconv(zscore(data(:,ii)),hrf./sigma(ii),num_iterations);
            end
        else
            for ii=1:nvar
                hrf=hrfa_TR(:,ii);
                data_deconv(:,ii) = rsHRF_iterative_wiener_deconv(zscore(data(:,ii)),hrf./sigma(ii),num_iterations);
            end
        end

    else
        disp('^.^') % do not perform deconvolution
    end
    toc
end
 
if ~flag_ROI % 3D volume/ 2D surface data
    if flag_nii_gii==1
        v0 = v(1);   v0.dt = [16,0]; v0.n = [1 1];
        ext_nii_gii = '.nii';
    else
        v0 = v;
        ext_nii_gii = '.gii';
    end
    HRF_para_str = {'Height', 'Time2peak','FWHM'};
    dat= zeros(v0.dim);
    if job.savedata.hrfnii_save
        %write nifti maps of the three HRF parameters
        for i=1:3
            fname = fullfile(outdir,[job.prefix,name,'_',HRF_para_str{i},ext_nii_gii]);
            eval(['out.',HRF_para_str{i},'{1} =','''',fname,''';'])
            dat(smask_ind)=PARA(i,:);
            rsHRF_write_file(fname,dat,flag_nii_gii,v0)
        end
        
        % write height PSC
        fname = fullfile(outdir,[job.prefix,name,'_Height_PSC',ext_nii_gii]);
        out.Height_PSC{1} = fname;
        dat(smask_ind)=PARA(1,:)./hrf_baseline;
        rsHRF_write_file(fname,dat,flag_nii_gii,v0)
        
        % write number of events
        fname = fullfile(outdir,[job.prefix,name,'_event_number',ext_nii_gii]);
        out.event_number{1} = fname;
        dat(smask_ind)=event_number;
        rsHRF_write_file(fname,dat,flag_nii_gii,v0)
    end
    if job.savedata.hrfmat_save
        save(fullfile(outdir,[job.prefix,name, '_hrf.mat']), 'smask_ind','v0','-append')
    end
    
    if job.rmoutlier && flag_nii_gii %only for 3D data
        PARA_rm = PARA;
        pvalue_rm=job.para_global.pvalue_rm;
        Inpainted_method =job.para_global.Inpainted ;
        for i=1:3
            fname = fullfile(outdir,[job.prefix,name,'_Olrm_',HRF_para_str{i},ext_nii_gii]);
            eval(['out.Olrm_',HRF_para_str{i},'{1}=','''',fname,''';'])
            v0.fname = fname;
            dat(smask_ind)=PARA(i,:);
            if i==1
                id = (PARA_rm(1,:)<=0); %Be Careful
                PARA_rm(1,id)=nan;
                [PARA_rm(1,:),idx,outliers] = rsHRF_deleteoutliers(PARA_rm(1,:),pvalue_rm,1);
                fprintf('Response height outlier: negative/zeros #%d + extreme #%d, [%3.3f %3.3f];\n',nnz(id),length(idx),min(outliers),max(outliers));
                dat(smask_ind)=PARA_rm(1,:);
                dat=rsHRF_inpaint_nans3(dat,Inpainted_method);
                PARA_rm(1,:) = dat(smask_ind);
                id_rm = union(find(id),idx);
                dat2 = zeros(v0.dim);
                PARA_rm0=zeros(1,size(PARA,2));
                PARA_rm0(1,id_rm)=1;
                dat2(smask_ind)=PARA_rm0;
                v00 = v0;
                fname = fullfile(outdir,[job.prefix,name,'_outlier_NAN',ext_nii_gii]);
                out.outlier{1} = fname; 
                v00.fname = fname; 
                spm_write_vol(v00,dat2);  clear dat2
            else
                PARA_rm(i,id_rm)=nan;
                dat(smask_ind)=PARA_rm(i,:);
                dat=rsHRF_inpaint_nans3(dat,Inpainted_method);
                PARA_rm(i,:) = dat(smask_ind);
            end
            if job.savedata.hrfnii_save
                spm_write_vol(v0,dat);
            end
        end
        if job.savedata.hrfnii_save || job.savedata.hrfmat_save
            % write number of events
            fname = fullfile(outdir,[job.prefix,name,'_Olrm_event_number',ext_nii_gii]);
            out.Olrm_event_number{1} = fname;
            v0.fname = fname;
            event_number_rm  = event_number;
            event_number_rm(id_rm)=nan;
            dat(smask_ind)=event_number_rm;
            dat=rsHRF_inpaint_nans3(dat,Inpainted_method);
            event_number_rm = dat(smask_ind);
            if job.savedata.hrfnii_save
                spm_write_vol(v0,dat);
            end
            
            if flag_save_psc
                %write height PSC
                fname = fullfile(outdir,[job.prefix,name,'_Olrm_Height_PSC',ext_nii_gii]);
                out.Olrm_Height_PSC{1} = fname;
                v0.fname = fname;
                HeightPSC_rm  = PARA_rm(1,:)./hrf_baseline;
                HeightPSC_rm(HeightPSC_rm<=0)=nan;
                [HeightPSC_rm,idx,outliers] = rsHRF_deleteoutliers(HeightPSC_rm,pvalue_rm,1);
                fprintf('Response height (PSC) outlier: negative/zeros #%d + extreme #%d, [%3.3f %3.3f];\n',nnz(id),length(idx),min(outliers),max(outliers));
                dat(smask_ind)=HeightPSC_rm;
                dat=rsHRF_inpaint_nans3(dat,Inpainted_method);
                HeightPSC_rm = dat(smask_ind);
                if job.savedata.hrfnii_save
                    spm_write_vol(v0,dat);
                end
            end
            
            if job.savedata.hrfmat_save
                if flag_save_psc
                    save(fullfile(outdir,[job.prefix,name, '_hrf.mat']), 'PARA_rm','event_number_rm','HeightPSC_rm','id_rm','-append')
                else
                    save(fullfile(outdir,[job.prefix,name, '_hrf.mat']), 'PARA_rm','event_number_rm','id_rm','-append')
                end
            end
        end
    end
    
    if ~isempty(connroinfo) && any(conndata_flag~=2)
        conid = [conndata_flag~=2];
        data_nD = nan(Nscans, prod(v0.dim)); 
        data_nD(:,smask_ind) = data;
        fprintf('BOLD Connectivity Analysis (%s)...\n',datetime(now,'ConvertFrom','datenum')); tic
        rsHRF_conn_run(data_nD, connroinfo(conid,:),v0,name,outdir,flag_pval_pwgc,flag_nii_gii);  
        fprintf('Done! Elapsed time is %ss\n',num2str(toc)); 
        clear data_nD data
    end
 
 
    if job.savedata.deconv_save || any(conndata_flag~=1)
        % writing back deconvolved data into nifti/gifti file
        fname = fullfile(outdir,[job.prefix,name,ext_nii_gii]);
        out.deconv_data{1} = fname;
        if flag_nii_gii==1
            v1 = v; dat3 = zeros(v1(1).dim);
            for i=1:Nscans
                v1(i).fname = fname;
                v1(i).dt = [16,0]; 
                dat3(smask_ind) = data_deconv(i,:);
                if job.savedata.deconv_save
                    spm_write_vol(v1(i),dat3);
                end
            end
        else
            dat_surf= gifti;
            dat = zeros(nvar_data,Nscans);
            dat(smask_ind,:) = data_deconv';
            dat_surf.cdata = dat;
            save(dat_surf,fname,'Base64Binary');
        end
            
 
        if ~isempty(connroinfo) && any(conndata_flag~=1)
            conid =  [conndata_flag~=1];
            data_deconv3D = nan(Nscans, prod(v0.dim)); 
            data_deconv3D(:,smask_ind) = data_deconv;
            name2 = [name,'_deconv'];
            fprintf('Deconvolved BOLD Connectivity Analysis (%s)...\n',datetime(now,'ConvertFrom','datenum')); tic
            rsHRF_conn_run(data_deconv3D, connroinfo(conid,:),v0,name2,outdir,flag_pval_pwgc,flag_nii_gii);                            
            fprintf('Done! Elapsed time is %ss\n',num2str(toc)); 

        end
 
        if job.rmoutlier && job.para_global.rmoutlier_deconv 
            if job.HRFE.hrfdeconv<3
                v1 = v; dat3 = zeros(v1(1).dim);
                fname = fullfile(outdir,[job.prefix,name,'_Olrm',ext_nii_gii]);
                out.Olrm_deconv_data{1} = fname;
                for i=1:Nscans
                    v1(i).fname = fname;
                    v1(i).dt = [16,0]; 
                    data_deconv_rm= data_deconv(i,:);
                    data_deconv_rm(id_rm)=nan;
                    dat3(smask_ind) = data_deconv_rm;
                    dat3=rsHRF_inpaint_nans3(dat3,Inpainted_method);
                    data_deconv3D(i,:) = dat3(:);
                    if job.savedata.deconv_save
                        spm_write_vol(v1(i),dat3);
                    end
                end
            end
            if ~isempty(connroinfo)&& any(conndata_flag~=1)
                conid =  [conndata_flag~=1];
                name2 = [name,'_deconv_Olrm'];
                fprintf('(Outlier removed) Deconvolved BOLD Connectivity Analysis (%s)...\n',datetime(now,'ConvertFrom','datenum')); tic
                rsHRF_conn_run(data_deconv3D, connroinfo(conid,:),v0,name2,outdir,flag_pval_pwgc,flag_nii_gii);     
                fprintf('Done! Elapsed time is %ss\n',num2str(toc)); 
                clear data_deconv3D
            end
 
            clear data_deconv_rm
        end
        clear data_deconv data
    end
    clear dat
else %ROI-wise
 
    if ~isempty(job.connectivity)
        fprintf('Connectivity analysis...\n');
        [connroinfo,conndata_flag] = rsHRF_conn_check(job);
        if any(conndata_flag~=2)
            conid = [conndata_flag~=2];
            rsHRF_conn_run(data, connroinfo(conid,:),[],name,outdir,flag_pval_pwgc);  
        end
        if any(conndata_flag~=1)
            conid = [conndata_flag~=1];
            name2 = [name,'_deconv'];
            rsHRF_conn_run(data_deconv, connroinfo(conid,:),[],name2,outdir,flag_pval_pwgc);   
        end
    end
    
    if job.savedata.deconv_save
        try
            save(fullfile(outdir,[job.prefix,name, '.mat']),'data','data_deconv','-append');
        catch
            save(fullfile(outdir,[job.prefix,name, '.mat']),'data','data_deconv','-v7.3');
        end
    end
    clear data data_deconv
    out=[];
end
if job.savedata.job_save
    save(fullfile(outdir,[job.prefix,name, '_job.mat']), 'job');  
end
