clc; %clears the command window and homes the cursor; see by typing "edit clc" in the MATLAB Command Window
clear; %clears variables and functions from memory; see by typing "edit clear" in the MATLAB Command Window

%% directory to rsHRF/code/ toolbox in "Downloads" folder
GHrepo_dir = fileparts(which('rsHRF_install_SPM.m'));
rsHRF_file = { %which files do you need?  
'wgr_get_parameters.m'  
'wgr_rshrf_estimation_canonhrf2dd_par2.m' %added
'wgr_rsHRF_FIR.m' %added
'rsHRF_estimation_temporal_basis.m'
'rsHRF_estimation_FIR.m'
'rsHRF_estimation_impulseest.m'
'rsHRF_mvgc.m'
'rsHRF_viewer.m'
'wgr_rsHRF_global_para.m'
'rsHRF.m'
'rsHRF_install_SPM.m'
%'rsHRF_toolbox.pptx' %rsHRF/documentation/
'rsHRF.man'
'rsHRF_logo.png' %additional to rsHRF/code/
'spm_rsHRF.m'
'tbx_cfg_rsHRF.m'
%'LICENSE' %rsHRF/
'deleteoutliers.m'
'inpaint_nans3.m'
'knee_pt.m'
'rest_nextpow2_one35.m' %added
'rest_IdealFilter.m' %added
%'README.md' 
%'update_log.txt' %rsHRF/code/history/
%'demo_jobs.zip' %rsHRF/documentation/demo/
};
%number of files
numfile = length(rsHRF_file);

%% directory to rsHRF toolbox in "spm12" folder
toolbox_dir = fullfile(spm('Dir'),'toolbox','rsHRF');

%toolbox already installed?
%NO
if ~exist(toolbox_dir,'dir')
    mkdir(toolbox_dir); %create path/to/spm12/toolbox/rsHRF folder
%YES
else 
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
