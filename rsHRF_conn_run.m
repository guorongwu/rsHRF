function rsHRF_conn_run(data, connroinfo,v0,name,outdir,flag_pval_pwgc,flag_nii_gii);
%data: nobs x nvar (3D index)
fprintf('Connectivity analysis...\n ')
meastr = {'pwGC','CGC','PCGC','Pearson','PartialPearson','Spearman','PartialSpearman',};

para_global = rsHRF_global_para;
regmode = para_global.regmode; % for GC
for j=1:size(connroinfo,1)
    fprintf('Conn %d\n',j)
    roiid = connroinfo{j,1};  nroi = size(roiid,1);
    flag_seedROI  = connroinfo{j,2};
    conn_type = connroinfo{j,3};
    order = connroinfo{j,4};
    ndinfo = connroinfo{j,5}; 
    ROI = [];
    for i=1:nroi
        tmp = nanmean(data(:,roiid{i,1}),2) ;
        ROI(:,i) = tmp;
    end
    if nroi==0 %&& isempty(v0)
        ROI = data;
    end
    if ~flag_seedROI % seed map
        if flag_nii_gii==1
            ext_nii_gii = '.nii';
        else
            ext_nii_gii = '.gii';
        end
            
        smask_ind = find(var(data)>0);
        dat.data = data(:,smask_ind);
        dat.ROI = ROI;
        dat3 = zeros(v0.dim);
        if conn_type==1 || conn_type==2  || conn_type==3 
            if conn_type==2
                conn_type = 1; % pairwise       
                warning('Change Contional GC to Pairwise GC')
            elseif conn_type==3
                conn_type = 1; % pairwise       
                warning('Change Partially Contioned GC to Pairwise GC')
            end
            disp('Pairwise GC for seed based connectivity analysis')            
            [M] = wgr_GC(dat,order,ndinfo,conn_type,regmode,flag_pval_pwgc);
            ordeinfo = ['_order',num2str(order)];
            for i=1:nroi
                if nroi>1
                    tmp = [num2str(i),'_'];
                else
                    tmp = '';
                end
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_outflow_pwGC',ordeinfo,ext_nii_gii]);
                out.outflow_pwGC{i} = fname;
                gc = M.GC_Matrix_Out(i,:);
                dat3(smask_ind) = gc;
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_outflow_N_pwGC',ordeinfo,ext_nii_gii]);
                out.outflow_N_pwGC{i} = fname;
                dat3(smask_ind) = wgr_pwgc_2normal(gc,M.nobs,order);
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                if flag_pval_pwgc
                    fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_outflow_pval_pwGC',ordeinfo,ext_nii_gii]);
                    out.outflow_pval_pwGC{i} = fname;
                    dat3(smask_ind) = M.pval_GC_Matrix_Out(i,:);
                    rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                end
                
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_inflow_pwGC',ordeinfo,ext_nii_gii]);
                out.inflow_pwGC{i} = fname;
                gc = M.GC_Matrix_In(i,:);
                dat3(smask_ind) = gc;
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_inflow_N_pwGC',ordeinfo,ext_nii_gii]);
                out.inflow_N_pwGC{i} = fname;
                dat3(smask_ind) = wgr_pwgc_2normal(gc,M.nobs,order);
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                seed_information = roiid(i,:);
                save(fullfile(outdir,[connroinfo{j,6},tmp,name,'_SeedInfo_pwGC',ordeinfo,'.mat']),'seed_information');
                
                if flag_pval_pwgc
                    fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_inflow_pval_pwGC',ordeinfo,ext_nii_gii]);
                    out.inflow_pval_pwGC{i} = fname;
                    dat3(smask_ind) = M.pval_GC_Matrix_In(i,:);
                    rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                end
            end
        else
            [M] = wgr_FC(dat,conn_type);
            for i=1:nroi
                if nroi>1
                    tmp = [num2str(i),'_'];
                else
                    tmp = '';
                end
                
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_corr_',meastr{connroinfo{j,3}},ext_nii_gii]);
                out.corr{i} = fname;
                dat3(smask_ind) = M.Matrix_r(i,:);
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                fname = fullfile(outdir,[connroinfo{j,6},tmp,name,'_Z_',meastr{connroinfo{j,3}},ext_nii_gii]);
                out.Z{i} = fname;
                dat3(smask_ind) = M.Matrix_z(i,:);
                rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
                
                seed_information = roiid(i,:);
                save(fullfile(outdir,[connroinfo{j,6},tmp,name,'_SeedInfo_',meastr{connroinfo{j,3}},'.mat']),'seed_information');

            end
        end
        
    else %ROI to ROI
        dat.data = ROI;
        dat.ROI = [];
        if conn_type==1 || conn_type==2 || conn_type==3 % 1:pairwise  2:conditional 3: partially conditioned
            M = wgr_GC(dat,order,ndinfo, conn_type,regmode, flag_pval_pwgc);
            save(fullfile(outdir,[connroinfo{j,6},name,'_',meastr{connroinfo{j,3}},'.mat']),'M','roiid');
        else
            M = wgr_FC(dat,conn_type);
            save(fullfile(outdir,[connroinfo{j,6},name,'_Corr_',meastr{connroinfo{j,3}},'.mat']),'M','roiid');
        end
        
    end 
end


function c = wgr_pwgc_2normal(gc,nobs,order)
c = (nobs-order).*gc - (order-1)/3;
c(c<0)=0;
c = sqrt(c);

function [M] = wgr_GC(dat,order,ndinfo,flag_pw_cgc,regmode,flag_pval_pwgc,m);
% flag_pw_cgc, 1: pairwise GC,  2: conditional GC,  3: partially conditioned GC.
gcstr = {'Pairwise GC','Conditional GC','Partially Conditioned GC'};
data = dat.data;
[nobs,nvar] = size(data);
ROI =  dat.ROI;
[nobsROI,nROI] = size(ROI);
M.seed_num = nROI;
if nROI
    if nobsROI~=nobs
        error('different observations (ROI vs Data)')
    end
end
if nobs<10
    warning('Too few observations !')
end

fprintf('Data dimension: %d x %d\n',nobs,nvar);
if flag_pw_cgc==2 % CGC
    if nobs<nvar
        fprintf('#observation < #variable\n');
        error('CGC stop!')
    end
end

if nargin<7
    m = 5000; %block size
end

M.nvar=nvar;
M.nobs=nobs;
M.order = order;
M.GC_type = gcstr{flag_pw_cgc};
M.ndinfo = ndinfo;
if flag_pw_cgc==3 % if nnz(ndinfo)==1
    nd = ndinfo(1);
    if isnan(nd)
        nd = 6;
    end
    ndmax = ndinfo(2); 
    if isnan(ndmax)
        ndmax = nd+1;
    end
end

nbin = ceil(nvar/m);
indX={}; indY={};
matrix={};
matrix_out = {};
matrix_in={};
p_matrix_out={};
p_matrix_in={};

if nROI %ROI to data
    nbinROI = ceil(nROI/m);
    for i=1:nbinROI
        if i~=nbinROI
            ind_X = (i-1)*m+1:i*m ;   
        else
            ind_X = (i-1)*m+1:nROI ;
        end
        indX{i} = ind_X; 
        for j=1:nbin
            if j~=nbin
                ind_Y = (j-1)*m+1:j*m ;
            else
                ind_Y = (j-1)*m+1:nvar ;
            end
            indY{j} = ind_Y; 
            if flag_pval_pwgc
                [matrix_out{i,j}, matrix_in{i,j}, p_matrix_out{i,j}, p_matrix_in{i,j}] = ...
                    wgr_seedGC(ind_X,ind_Y,ROI,data,order,1,regmode); 
            else
                [matrix_out{i,j}, matrix_in{i,j}] = wgr_seedGC(ind_X,ind_Y,ROI,data,order,0,regmode); %only pairwise.
            end
        end
    end
else % data to data
    if flag_pw_cgc==3 
        [y_inform_gain, cind] = wgr_init_partial_conditioning(data,[],ndmax,order);
        M.information_gain = y_inform_gain;
        M.condition_id = cind;
    end
    if flag_pw_cgc==1
        [M.GC_Matrix, M.pval_Matrix] = wgr_PWGC(data,order,regmode,flag_pval_pwgc);
    end
    if flag_pw_cgc==2
        [M.GC_Matrix, M.pval_Matrix] = rsHRF_mvgc(data',order,regmode,0,flag_pval_pwgc);
    end
    
    if flag_pw_cgc==3
        [M.GC_Matrix, M.pval_Matrix] = wgr_PCGC(data,order,cind,nd,regmode,flag_pval_pwgc);
    end
end

if nROI
    M.GC_Matrix_Out = nan(nROI,nvar);
    M.GC_Matrix_In = M.GC_Matrix_Out ;
    for i=1:nbinROI
        for j=1:nbin
            M.GC_Matrix_Out(indX{i},indY{j}) =  matrix_out{i,j};
            M.GC_Matrix_In(indX{i},indY{j}) =  matrix_in{i,j};
            if flag_pval_pwgc
                M.pval_GC_Matrix_Out(indX{i},indY{j}) =  p_matrix_out{i,j};
                M.pval_GC_Matrix_In(indX{i},indY{j}) =  p_matrix_in{i,j};
            end
        end
    end 

else
    
    if flag_pw_cgc==1
        M.GC_Matrix_N = wgr_pwgc_2normal(M.GC_Matrix,nobs,order);
    end        
end
clear matrix* p_* indX indY

function [F,pvalue] = wgr_PWGC(data,order,regmode,flag_pval);

[nvar] = size(data,2);
F = zeros(nvar);
if flag_pval
    pvalue = nan(nvar);
end
for drive=1:nvar
    for target=1:nvar
        if drive~=target
            dat = data(:,[drive target])';
            [F0,p0] = rsHRF_mvgc(dat,order,regmode,0,flag_pval);
            F(drive,target) = F0(1,2);
            pvalue(drive,target)  = p0(1,2);
            F(target,drive) = F0(2,1);
            pvalue(target,drive)  = p0(2,1);
        end
    end
end

function [F,pvalue] = wgr_PCGC(data,order,cind,nd,regmode,flag_pval);
[nvar] = size(data,2);
F = zeros(nvar);
if flag_pval
    pvalue = nan(nvar);
end
parfor drive=1:nvar
    for target=1:nvar
        if drive~=target
            zid = setdiff(cind(drive,:),target,'stable');
            dat = data(:,[drive target zid(1:nd)])';
            [F(drive,target),pvalue(drive,target)] = ...
                rsHRF_mvgc(dat,order,regmode,1,flag_pval);
        end
    end
end

function [gc_out, gc_in, p_out, p_in] = wgr_seedGC(ind_X,ind_Y,ROI,data1,order,flag_pval,regmode);

[nvar1] = length(ind_X);
[nvar2] = length(ind_Y);
gc_out = zeros(nvar1,nvar2);
gc_in = gc_out;
if flag_pval
    p_out = nan(nvar1,nvar2);
    p_in = p_out;
end
parfor drive=1:nvar1
    for target=1:nvar2
        data = [ROI(:,ind_X(drive)) data1(:,ind_Y(target))]';
        [F,pvalue] = rsHRF_mvgc(data,order,regmode,0,flag_pval);
        if length(F)>1
            gc_out(drive,target) = F(1,2); 
            gc_in(drive,target)  = F(2,1); 

            p_out(drive,target)  = pvalue(1,2);
            p_in(drive,target)   = pvalue(2,1); 
        end
    end
end

function [M] = wgr_FC(dat,conn_type,m);
% conn_type, 4/5: pearson,  6/7: spearman
if conn_type==4
    con_type =  'Pearson'; flag_partial=0;
elseif conn_type==5
    con_type = 'Pearson'; flag_partial=1;
elseif conn_type==6
    con_type = 'Spearman'; flag_partial=0;
elseif conn_type==7
    con_type = 'Spearman'; flag_partial=1;
end

data = dat.data;
[nobs,nvar] = size(data);
ROI =  dat.ROI;
[nobsROI,nROI] = size(ROI);
M.seed_num = nROI;
if nROI
    if nobsROI~=nobs
        error('different observations (ROI vs Data)')
    end
end
if nobs<10
    warning('Too few observations !')
end

fprintf('Data dimension: %d x %d\n',nobs,nvar);

if nargin<3
    m = 5000; %block size
end

M.nvar=nvar;
M.nobs=nobs;

nbin = ceil(nvar/m);
indX={}; indY={};
matrix={};

if nROI %ROI to data
    nbinROI = ceil(nROI/m);
    for i=1:nbinROI
        if i~=nbinROI
            ind_X = (i-1)*m+1:i*m ;   
        else
            ind_X = (i-1)*m+1:nROI ;
        end
        indX{i} = ind_X; 
        for j=1:nbin
            if j~=nbin
                ind_Y = (j-1)*m+1:j*m ;
            else
                ind_Y = (j-1)*m+1:nvar ;
            end
            indY{j} = ind_Y;  
            matrix_r{i,j} = corr(ROI(:,ind_X),data(:,ind_Y), 'type', con_type); 
            matrix_z{i,j} = atanh(matrix_r{i,j}) ;
        end
    end
else % data to data
    if flag_partial % partial correlation
        if nobs<nvar
            fprintf('#observation < #variable\n');
            error('partial correlation stop!')
        end
    end
    if ~flag_partial
        for i=1:nbin
            if i~=nbin
                ind_X = (i-1)*m+1:i*m ;   
            else
                ind_X = (i-1)*m+1:nvar ;
            end
            indX{i} = ind_X; 
            for j=1:nbin
                if j~=nbin
                    ind_Y = (j-1)*m+1:j*m ;
                else
                    ind_Y = (j-1)*m+1:nvar ;
                end
                indY{j} = ind_Y; 
                [matrix_r{i,j},matrix_p{i,j}] = corr(data(:,ind_X),data(:,ind_Y), 'type',con_type); 
                matrix_z{i,j} = atanh(matrix_r{i,j}) ;
            end
        end
    else
        indX{1} = 1:nvar ; indY{1} = 1:nvar ;
        [matrix_r{1,1},matrix_p{1,1}] = partialcorr(data, 'type',con_type); 
        matrix_z{1,1} = atanh(matrix_r{1,1}) ;
    end
end

if nROI
    M.Matrix_r = nan(nROI,nvar);
    M.Matrix_z = M.Matrix_r;
    for i=1:nbinROI
        for j=1:nbin
            M.Matrix_r(indX{i},indY{j}) =  matrix_r{i,j};
            M.Matrix_z(indX{i},indY{j}) =  matrix_z{i,j};
        end
    end 
    
else
    M.Matrix_r = nan(nvar,nvar);
    for i=1:nbin
        for j=1:nbin
            M.Matrix_r(indX{i},indY{j}) =  matrix_r{i,j};
            M.Matrix_z(indX{i},indY{j}) =  matrix_z{i,j};
            M.Matrix_pval(indX{i},indY{j}) =  matrix_p{i,j};
        end
    end 
    M.Matrix_r(1:nvar+1:end)=0; %remove diag value.
    M.Matrix_z(1:nvar+1:end)=0; %remove diag value.
    M.Matrix_pval(1:nvar+1:end)=1; %remove diag value.
end
clear matrix* indX indY

function [y, ind] = wgr_init_partial_conditioning(data,seed_signal,ndmax,order)
[N,nvar] = size(data);
X=cell(nvar,1);
past_ind = repmat([1:order],N-order,1) + repmat([0:N-order-1]',1,order);
for i=1:nvar
    past_data= reshape(data(past_ind,i),N-order,order);
    X{i}= past_data - repmat(mean(past_data), N-order, 1); %%remove mean
end

ind=zeros(nvar,ndmax);
y=ind;

if ~isempty(seed_signal)
   [N2,nvar2] = size(seed_signal);
   if N~=N2
       error('check #observation of seed_signal !')
   else
       ind = nan(nvar2,ndmax);  y = ind;
       parfor drive=1:nvar2
           past_data = reshape(seed_signal(past_ind,drive),N-order,order);
           drive_sig = past_data - repmat(mean(past_data), N-order, 1);
           [y(drive,:), ind(drive,:)]=wgr_information_gain(drive_sig,X,nvar,ndmax);
       end
   end
else
    ind = nan(nvar,ndmax);  y = ind;
    parfor drive=1:nvar
        [y(drive,:), ind(drive,:)]=wgr_information_gain(drive,X,nvar,ndmax); %do not allow to dynamic plot
    end
end

function [y, ind]= wgr_information_gain(drive,X,nvar,ndmax)
if nnz(drive)==1 
    indici=setdiff(1:nvar,drive);%,'stable');
    t=X{drive};
else %seed signal
    indici= 1:nvar ;
    t=drive;
end

Zt=[]; ind=nan(1,ndmax); y = ind;
for nd=1:ndmax
    n1=length(indici);
    z=zeros(n1,1);
    for k=1:n1
        Zd=[Zt X{indici(k)}];
        z(k)= wgr_MI_gaussian(Zd,t);
    end
    [y(1,nd), id] = max(z);
    Zt = [Zt, X{indici(id)}];
    ind(1,nd) = indici(id);
    indici = setdiff(indici, indici(id));%,'stable');
end
 
function mi = wgr_MI_gaussian(Y,X)
% X: N x k_x  Y: N x k_y 
% condtional mutual information
% mi = gaussian mutual information X,Y
XY_cov = X'*Y; %%here we delete the factor: 1/N.
X_cov = X'*X;
Y_cov = Y'*Y;
all_cov = [X_cov XY_cov;XY_cov' Y_cov];
mi = 0.5*log(abs(det(X_cov)*det(Y_cov)/det(all_cov)));
return
