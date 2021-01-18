clc; clear; 

%% directory to rsHRF/code/ toolbox in "Downloads" folder
GHrepo_dir = fileparts(which('rsHRF_install_SPM.m'));
rsHRF_file = { %which files do you need?  
'tbx_cfg_rsHRF.m'  
'rsHRF.m'                            
'rsHRF.man'                          
'rsHRF_logo.png'    
'rsHRF_viewer.m'                     
'spm_rsHRF.m'      
'rsHRF_install_SPM.m'  
'rsHRF_estimation_FIR.m'             
'rsHRF_estimation_impulseest.m'      
'rsHRF_estimation_temporal_basis.m'  
'rsHRF_find_event_vector.m'          
'rsHRF_get_HRF_parameters.m'   
'rsHRF_deleteoutliers.m'                             
'rsHRF_inpaint_nans3.m'              
'rsHRF_iterative_wiener_deconv.m'    
'rsHRF_knee_pt.m'                   
'rsHRF_mvgc.m' 
'rsHRF_global_para.m'                
'rsHRF_ROI_sig_job.m'                
'rsHRF_band_filter.m'                
'rsHRF_check_ROI.m'                  
'rsHRF_conn_check.m'                 
'rsHRF_conn_run.m'   
'rsHRF_denoise_job.m'   
'rsHRF_conn_job.m'                   
'rsHRF_deconv_job.m' 
'rsHRF_read_GIfTI_job.m'             
'rsHRF_read_NIfTI_job.m'             
'rsHRF_write_file.m'                 
'rsHRF_update_log.txt'               
'demo_jobs.zip'  
%'README.md' 
%'LICENSE' %rsHRF/
};
%number of files
numfile = length(rsHRF_file);

%% directory to rsHRF toolbox in "spm12" folder
toolbox_dir = fullfile(spm('Dir'),'toolbox','rsHRF');

%toolbox already installed?

if ~exist(toolbox_dir,'dir')%NO
    mkdir(toolbox_dir); %create path/to/spm12/toolbox/rsHRF folder

else %YES
    try %zip the older version of the rsHRF toolbox
        zip(fullfile(toolbox_dir,['rsHRF',date,'.zip']),rsHRF_file,toolbox_dir)
        disp(['Your older version of the rsHRF toolbox has been compressed and can be accessed through rsHRF', date, '.zip.'])
    catch
        toolbox_dir_content = dir(toolbox_dir);
        if numel(toolbox_dir_content) <= 2 %rsHRF toolbox is empty
            fprintf(['Skippable Error:', '\n', 'When trying to zip the older version of the rsHRF toolbox, the path/to/spm12/toolbox/rsHRF folder was empty.', ...
            '\n', 'All files will be replaced by those in the newest version of the rsHRF toolbox.', '\n']);
        elseif numel(toolbox_dir_content) < numfile+2 %rsHRF toolbox is not complete
            fprintf(['Skippable Error:', '\n', 'When trying to zip the older version of the rsHRF toolbox, the path/to/spm12/toolbox/rsHRF folder was not complete.', ...
            '\n', 'All files will be replaced by those in the newest version of the rsHRF toolbox.', '\n']);
        else
            fprintf(['Undefined Error:', '\n', 'When trying to zip the older version of the rsHRF toolbox.', '\n']);
        end
    end
end

%% copy files: path/to/rsHRF/code/* --> path/to/spm12/toolbox/rsHRF/
for ifile=1:numfile
    file = fullfile(toolbox_dir,rsHRF_file{ifile});
    if exist(file,'file')
        delete(file)
    end
    copyfile(fullfile(GHrepo_dir,rsHRF_file{ifile}),file)
end

%unzip(fullfile(toolbox_dir,'demo_jobs.zip'),toolbox_dir) %rsHRF/documentation/demo/

%% done!
disp('Done, rsHRF');
%start Resting State HRF GUI
str='rsHRF'; disp(str); eval(str);
%close
pause(3);

%start Connectivity GUI
disp('Connectivity');
str2='rsHRF conn'; eval(str2);
pause(3);

%close both GUIs
close all;
