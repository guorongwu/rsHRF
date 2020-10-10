function [mat,atlas,nii,atlasmesh,gii]= rsHRF_check_ROI(ROI, funii, flag_delete)
mat = {};  atlas = {}; nii = {}; atlasmesh = {}; gii = {};
k1 =1 ; k2 = 1; k3 = 1; k4 = 1; k5 = 1;
for i=1:length(ROI)
    tmp  = ROI{i};
    flag = isfield(tmp,'ROI');
    if flag
        for j=1: size(tmp.ROI,1)
            tmp2 = tmp.ROI(j,:);
            xY = [];
            xY.def='sphere';
            xY.xyz=tmp2(1:3)';
            xY.spec=tmp2(4);
            [xY, XYZmm, id] = spm_ROI(xY, funii);
            if isempty(id)
                error(sprintf('\n\n====(ROI)Error Information====\nno voxel found in sphere with center (%3.1f, %3.1f, %3.1f) and radius %3.1f.\nTry to increase the radius???\n',tmp2));
            else
                fprintf('#%d voxels found in sphere with center (%3.1f, %3.1f, %3.1f) and radius %3.1f.\n',length(id),tmp2)
            end
            mat{k1,1} = id;
            mat{k1,2} = tmp2;
            mat{k1,3} = '';
            k1 = k1+1;
            continue;
        end
    end
        
    flag = isfield(tmp,'atlas');
    if flag
        for j=1: length(tmp.atlas)
            tmp2 = strcat(tmp.atlas{j});
            if ~isempty(tmp2)
                atlas{k2,1} = tmp2;
                niinew  = spm_file(tmp2,'prefix','bin_');
                spm_imcalc({funii;tmp2},niinew,'(i1~=0).*i2',{0,0,0,16});
                v = spm_vol(niinew);
                datam = spm_read_vols(v); datam(isnan(datam)) = 0;
                idu = unique(datam(:)); idu(idu==0)=[];
                for k=1:length(idu)
                    atlas{k2,1} = find(datam==idu(k)); 
                    atlas{k2,2} = tmp2; 
                    atlas{k2,3} = idu(k); 
                    k2 = k2+1;
                end            
                clear datam
                if flag_delete
                    delete(v.fname)
                end
            end
        end
        continue;
    end
    
    flag = isfield(tmp,'images');
    if flag
        for j=1: length(tmp.images)
            tmp2 = strcat(tmp.images{j});      
            if ~isempty(tmp2)
                nii{k3,1} = tmp2;
                niinew  = spm_file(tmp2,'prefix','bin_');
                spm_imcalc({funii;tmp2},niinew,'(i1~=0).*i2',{0,0,0,16});
                v  =spm_vol(niinew);
                datam = spm_read_vols(v);
                nii{k3,1} = find(datam); clear datam
                nii{k3,2} = tmp2; 
                nii{k3,3} = ''; 
                if flag_delete
                    delete(v.fname)
                end
                k3 = k3+1;
            end
        end
        continue;
    end
    
    flag = isfield(tmp,'meshmask');
    if flag
        for j=1: length(tmp.meshmask)
            tmp2 = strcat(tmp.meshmask{j});      
            if ~isempty(tmp2)
                gii{k4,1} = tmp2;
                datam = gifti(tmp2);
                gii{k4,1} = find(datam.cdata); clear datam
                gii{k4,2} = tmp2; 
                gii{k4,3} = ''; 
                k4 = k4+1;
            end
        end
        continue;
    end
    
    flag = isfield(tmp,'meshatlas');
    if flag
        for j=1: length(tmp.meshatlas)
            tmp2 = strcat(tmp.meshatlas{j});
            if ~isempty(tmp2)
                atlasmesh{k5,1} = tmp2;
                datam = gifti(tmp2);
                idu = unique(datam.cdata(:)); idu(idu==0)=[]; 
                for k=1:length(idu)
                    atlasmesh{k5,1} = find(datam.cdata==idu(k)); 
                    atlasmesh{k5,2} = tmp2; 
                    atlasmesh{k5,3} = idu(k); 
                    k5 = k5+1;
                end            
                clear datam
            end
        end
        continue;
    end
end
