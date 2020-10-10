function rsHRF_write_file(fname,dat3,flag_nii_gii,v0)
if flag_nii_gii==1
    v0.fname = fname;
    spm_write_vol(v0,dat3);
else
    dat_surf= gifti;
    dat_surf.cdata = dat3;
    save(dat_surf,fname,'Base64Binary');
end