function st = rsHRF_viewer(job);
% gronwu@gmail.com ;  Guo-Rong Wu
% 2017, 18th May.
% 2019, 25th Oct, updated.
% 2020, 9th Oct, colorbar Yticklabel removed.
underlay_img = job.underlay_nii{1};
img = job.stat_nii{1};
HRF_mat = job.HRF_mat;

clear st;          
spm_orthviews('Reset');
global st
st.fig = figure('Visible','on',...
              'numbertitle','off',...                                   
              'menubar','none',...
              'units','normalized',...
              'color','w',...
              'position',[0.05,0.05,0.25,0.4],...
              'name','HRF Viewer (v1.1)',...
              'resize','on');
          
h = spm_orthviews('Image', underlay_img, [0 0.05 1 1]);
st.h = h;
st.underlay_img  = underlay_img;

%% MNI input
info.mni_tx = uicontrol('Parent',st.fig,...
                    'style','pushbutton',...
                    'units','norm',...
                    'position',[0.58 0.45 0.21 0.08],...
                    'string','MNI (x,y,z)',...               
                    'backgroundc','w',...
                    'fontunits', 'normalized', 'fontsize',0.35,...                  
                    'horizontalalign','center',...                    
                    'value',1,...
                    'TooltipString','Sphere Radius'); 
                
info.mni_tx = uicontrol('Parent',st.fig,...
                    'style','pushbutton',...
                    'units','norm',...
                    'position',[0.82 0.45 0.13 0.08],...
                    'string','Radius',...               
                    'backgroundc','w',...
                    'fontunits', 'normalized', 'fontsize',0.35,...                  
                    'horizontalalign','center',...                    
                    'value',1,...
                    'TooltipString','Sphere Radius'); 
                
info.ed_mni = uicontrol('Parent',st.fig,...
                        'style','edit',...
                        'unit','norm',...
                        'position',[0.58 0.36 0.21 0.08],...
                        'fontunits', 'normalized', 'fontsize',0.5,...                                                  
                        'string','0 0 0',...
                        'KeyPressFcn',@wgr_edit_mni,...
                        'horizontalalign','center');            
                    
info.ed_radius = uicontrol('Parent',st.fig,...
                        'style','edit',...
                        'unit','norm',...
                        'position',[0.82 0.36 0.13 0.08],...
                        'fontunits', 'normalized', 'fontsize',0.5,...                                                
                        'string','0',...
                        'horizontalalign','center');            
                    
%% threshold /extent
info.thresh = uicontrol('Parent',st.fig,...
                    'style','pushbutton',...
                    'units','norm',...
                    'position',[0.58 0.25 0.21 0.08],...
                    'string','Threshold',...               
                    'backgroundc','w',...
                    'fontunits', 'normalized', 'fontsize',0.4,...                  
                    'horizontalalign','center',...                    
                    'value',1,...
                    'TooltipString','...'); 
                
info.extent = uicontrol('Parent',st.fig,...
                    'style','pushbutton',...
                    'units','norm',...
                    'position',[0.82 0.25 0.13 0.08],...
                    'string','Extent',...               
                    'backgroundc','w',...
                    'fontunits', 'normalized', 'fontsize',0.4,...                  
                    'horizontalalign','center',...                    
                    'value',1,...
                    'TooltipString','Sphere Radius'); 
                
info.ed_threshold = uicontrol('Parent',st.fig,...
                        'style','edit',...
                        'unit','norm',...
                        'position',[0.58 0.15 0.21 0.08],...
                        'fontunits', 'normalized', 'fontsize',0.5,...                                   
                        'string','-1.9 1.9',...
                        'callback',@wgr_update,...
                        'horizontalalign','center');            
info.ed_extent = uicontrol('Parent',st.fig,...
                        'style','edit',...
                        'unit','norm',...
                        'position',[0.82 0.15 0.13 0.08],...
                        'fontunits', 'normalized', 'fontsize',0.5,...                                                  
                        'string','20',...
                        'callback',@wgr_update,...
                        'horizontalalign','center');   
                    
info.intensity = uicontrol('Parent',st.fig,...
                  'style','pushbutton',...
                  'units','norm',...
                  'backgroundc','w',...
                  'position',[0.58 0.05 0.21 0.08],...
                  'string','',...                
                  'TooltipString','value/intensity',...
                  'fontunits', 'normalized', 'fontsize',0.5,'val',1); 
              
info.plot = uicontrol('Parent',st.fig,...
                  'style','pushbutton',...
                  'units','norm',...
                  'backgroundc','w',...
                  'position',[0.82 0.05 0.13 0.08],...
                  'string','Plot',...                
                  'callback',@wgr_plot_hrf,...
                  'TooltipString','Plot HRF shape',...
                  'fontunits', 'normalized', 'fontsize',0.4,'val',1); 

st.info = info;

if ~isempty(img)
    vv = spm_vol(img);
    st.v = vv;
    refinfo = load(HRF_mat{1}{1},'v0');
    if any(vv.mat - refinfo.v0.mat)
        fprintf('(Underlay image) %s \n is different from (1st, ''v0.mat'') \n%s\n',underlay_img, HRF_mat{1}{1})
    end
    [data, mni] = spm_read_vols(vv);
    data(isnan(data))=0;
    st.mni = mni;
    st.stat_data = data;
    [peakcoord, peakintensity] = wgr_update;
   
end
spm_orthviews('Reposition',peakcoord);
movegui(st.fig,'center');
set(info.ed_mni,'string',sprintf('[%d,%d,%d]',round(peakcoord)));
set(info.intensity,'string',sprintf('%3.3f',peakintensity));

%% HRF shapes
fprintf('checking HRF mat-files...\n')
refinfo = load(HRF_mat{1}{1},'v0','smask_ind');
if 0 % keep only survived voxels
    idunique = intersect(refinfo.smask_ind,pos);
else
    idunique = refinfo.smask_ind;
end
k00=0;
for i=1:length(HRF_mat)
    group = HRF_mat{i};
    for j=1:size(group,1)
        aa = load(group{j},'v0','smask_ind');
        if any(aa.v0.mat - refinfo.v0.mat)
            fprintf('%s \n is different from (1st, ''v0.mat'') \n%s\n',group{j}, HRF_mat{1}{1})
        else
            idunique = intersect(idunique,aa.smask_ind);
        end
        k00=k00+1;
    end
end
fprintf('\nDone\n')
% fprintf('keep only survived voxels\n other voxels information will be removed.\n')
% fprintf('Loading HRF Shapes (only survived #%d voxels)...\n',length(idunique))
fprintf('Loading HRF Shapes...\n',length(idunique))

ng = length(HRF_mat);
aa = load(HRF_mat{1}{1},'hrfa');
mf = zeros(size(aa.hrfa,1),length(idunique),ng);
sf = mf;

for i=1:ng
    group = HRF_mat{i};
    k00 = size(group,1);
    for j=1:k00
        fprintf('.')
        aa = load(group{j},'v0','smask_ind','hrfa');
         if  j==1
            HRFa = zeros(size(aa.hrfa,1),length(idunique),k00); 
        end
        [C,ia,ib] = intersect(aa.smask_ind, idunique);
        HRFa(:,:,j) = aa.hrfa(:,ia);
    end
    
    mf(:,:,i) = mean(HRFa,3);
%     sf(:,:,i)  = std(HRFa,0,3);%std
    sf(:,:,i)  = std(HRFa,0,3)./sqrt(k00);%SE
end
fprintf('Done\n')
aa = load(group{j},'para');
st.dt = aa.para.dt;
st.mf = mf;
st.sf = sf;
st.idunique  = idunique;
set(st.fig,'WindowButtonDownFcn',@wgr_get_crosshairs); %double click

function [] = wgr_edit_mni(varargin)
global st
mni_coord = str2num(get(st.info.ed_mni,'string'));
spm_orthviews('Reposition',mni_coord);
v = st.v;
data = st.stat_data;
c_cor = mni2cor(mni_coord, v.mat)';
cvalue = data(c_cor(1),c_cor(2),c_cor(3));
set(st.info.intensity,'string',sprintf('%3.3f',cvalue));

    
function [] = wgr_get_crosshairs(varargin)
global st

if strcmp(get(gcf,'SelectionType'),'normal')
    mni_coord= spm_orthviews('pos')';
    v = st.v;
    data = st.stat_data;
    c_cor = mni2cor(mni_coord,v.mat);
    cvalue = data(c_cor(1),c_cor(2),c_cor(3));
    set(st.info.ed_mni,'string',sprintf('[%d,%d,%d]',round(mni_coord)));
    set(st.info.intensity,'string',sprintf('%3.3f',cvalue));
    return
end


function []=wgr_plot_hrf(varargin)
global st
mni_coord= spm_orthviews('pos')';
radius = str2num(get(st.info.ed_radius,'string'));
if radius
    xY = [];
    xY.def='sphere';
    xY.xyz= mni_coord';
    xY.spec = radius;
    [xY, XYZmm, ~] = spm_ROI(xY, st.tmpmni);
    mni_coord = [XYZmm';mni_coord];
else
    xY.xyz= mni_coord';
end

mf = st.mf;
sf = st.sf;
idunique = st.idunique;
v = st.v;
dt = st.dt;
data = st.stat_data;
c_cor = mni2cor(mni_coord,v.mat);
cid = sub2ind(size(data),c_cor(:,1),c_cor(:,2),c_cor(:,3) );
% id = find(idunique==cid); %length(cid)==1;
[C,id,ib] = intersect(idunique,cid);

mf0 = squeeze(mean(mf(:,id,:),2));
sf0  = squeeze(mean(sf(:,id,:),2));

nobs = size(mf,1);
xx = (1:nobs)*dt;

if isempty(id)
    fprintf('no HRF information in this position \n')
else
    figure1 = figure('color','w'); 
    ax = axes('Parent',figure1,'Position',[0.15 0.15 0.7 0.7]);
    movegui(figure1,'east');

    ng  = size(mf0,2);
    if ng<7
        colo = {'r','b','g',[1 0 1],'k',[0.8 0.6 0]};
    else
        colo1 = jet(ng);
        for i=1:ng
            colo{i} = colo1(i,:);
        end
    end
    
    hold all;
    for i=1:ng
        flag_norm = 0; %peak normalization
        if flag_norm
            mf0(:,i)  = mf0(:,i) ./max(mf0(:,i) );
        end
        plot(ax, xx,mf0(:,i) ,'LineWidth',2,'color',colo{i});
    end
    for i=1:ng
        if any(sf0(:,i))
            fill(ax, [xx fliplr(xx)],  [mf0(:,i)'+sf0(:,1)' fliplr(mf0(:,i)'-sf0(:,i)')],colo{i}, ...
                'LineStyle','none','LineWidth',1,'EdgeColor',colo{i},'edgealpha',0,'facealpha',0.5);  
        end
    end
    xlim(ax,[0 max(xx)])
    axis square
    a = mf0 - 1.5*sf0 ; minx = min(a(:));
    a = mf0 + 1.5*sf0; maxx= max(a(:));
    ylim(ax,[minx maxx])
    hold off
    gg={};
    for i=1:ng
        gg{i} = ['G',num2str(i)];
    end
    legend(ax,gg,'box','off')
    set(ax,'FontName','Calibri','FontSize',14); %'Cambria','Times New Roman'
    box on
    xlabel(ax,'Time(s)')
    if  radius
        title(sprintf('HRF (center MNI:[%d,%d,%d], spherical radius: %3.2f,#%dvoxels)',round(xY.xyz), radius,size(mni_coord,1)));
    else
        title(sprintf('HRF (MNI:[%d,%d,%d])',round(xY.xyz)));
    end
%     set(st.info.ed_mni,'string',sprintf('[%d,%d,%d]',round(mni_coord)));
%     set(st.info.intensity,'string',sprintf('%3.3f',cvalue));
end
return


% function [peakcoord, peakintensity] = wgr_threshold_map()
function [peakcoord, peakintensity] = wgr_update(varargin)
mni_coord= spm_orthviews('pos')';
global st
mni = st.mni ;
data = st.stat_data ;
T_threhold = str2num(get(st.info.ed_threshold,'string')); 
extent_clu  = str2num(get(st.info.ed_extent,'string'));
v =  st.v ;

data(data>T_threhold(1)&data<T_threhold(2))=0; 
XYZ = mni2cor(mni', v.mat)';

if extent_clu>1
    id = find(data~=0);
    [x0,y0,z0] = ind2sub(v.dim,id);
    A = spm_clusters([x0 y0 z0]');
    clu = unique(A(:));
    clu_len = zeros(max(clu),1);
    id2 = [];
    for kk= 1:max(clu)
        tmp  = find(A(:)==kk);
        clu_len(kk) = length(tmp);
        if clu_len(kk)>= extent_clu
            id2 = [id2;tmp];
        end
    end
    dat = zeros(v.dim);
    dat(id(id2)) = data(id(id2));    
    data = dat; clear dat
end

pos = find(data~=0);
numVoxels = length(pos);
if numVoxels==0
    fprintf('No survived results  <%3.2f or >%3.2f \n',T_threhold)
    return
else
    fprintf('#%d voxels\n',numVoxels)
end
tmpmni = mni(:,pos);
tmpintensity = data(pos);
tmpXYZ = XYZ(:,pos);
st.tmpmni = tmpmni;
st.tmpXYZ = tmpXYZ;
st.tmpZ = tmpintensity;

peakpos = find(abs(tmpintensity) == max(abs(tmpintensity)));
peakpos = peakpos(1);
peakcoord = tmpmni(:,peakpos)';
peakintensity = tmpintensity(peakpos);

if any(tmpintensity>0) && any(tmpintensity<0) 
    spm_figure('Colormap','gray-jet')
elseif any(tmpintensity>0) && ~any(tmpintensity<0) 
    spm_figure('Colormap','gray-hot')
else %if  ~any(tmpintensity>0) & any(tmpintensity<0) 
    spm_figure('Colormap','gray-cool')
end

st.h = spm_orthviews('Image', st.underlay_img, [0 0.05 1 1]);
spm_orthviews('AddBlobs', st.h, st.tmpXYZ, st.tmpZ, st.v.mat);
spm_orthviews('Reposition',mni_coord);
fprintf('Peak MNI: [%d %d %d], Peak value: %2.3f\n',[round(peakcoord) peakintensity])
spm_orthviews('Redraw');
hh= cellfun(@(x) isempty(x), st.vols);
id= find(hh);
for i=2:id(1)-2
    st.vols{i, 1}.blobs{1, 1}.cbar.YTickLabel={};
end

return 

function coordinate = mni2cor(mni, T)
if isempty(mni)
    coordinate = [];
    return;
end
coordinate = [mni(:,1) mni(:,2) mni(:,3) ones(size(mni,1),1)]*(inv(T))';
coordinate(:,4) = [];
coordinate = round(coordinate);
return;
