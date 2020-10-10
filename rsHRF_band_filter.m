function  x = rsHRF_band_filter(x,TR,Bands,m)
% data: x  nobs*nvar
if nargin<4
    m = 5000; %block size
end
nvar = size(x,2);
nbin = ceil(nvar/m);
for i=1:nbin
    if i~=nbin
        ind_X = (i-1)*m+1:i*m ;   
    else
        ind_X = (i-1)*m+1:nvar ;
    end
    x1 = x(:,ind_X);
    x1 = conn_filter(TR,Bands,x1,'full') +...%mean-center by default
     repmat(mean(x1,1),size(x1,1),1);    %mean back in
    x(:,ind_X) = x1;
end
clear x1;

function [y,fy]=conn_filter(rt,filter,x,option)
% from CONN toolbox, www.conn-toolbox.org
USEDCT=true;
if nargin<4, option='full'; end
if strcmpi(option,'base'), Nx=x; x=eye(Nx); end
if USEDCT % discrete cosine basis
    Nx=size(x,1);
    fy=fft(cat(1,x,flipud(x)),[],1);
    f=(0:size(fy,1)-1);
    f=min(f,size(fy,1)-f);
    switch(lower(option))
        case {'full','base'}
            idx=find(f<filter(1)*(rt*size(fy,1))|f>=filter(2)*(rt*size(fy,1)));
            %idx=idx(idx>1);
            fy(idx,:)=0;
            k=1; %2*size(fy,1)*(min(.5,filter(2)*rt)-max(0,filter(1)*rt))/max(1,size(fy,1)-numel(idx));
            y=real(ifft(fy,[],1))*k;
            y=y(1:Nx,:);
        case 'partial'
            idx=find(f>=filter(1)*(rt*size(x,1))&f<filter(2)*(rt*size(x,1)));
            %if ~any(idx==1), idx=[1,idx]; end
            y=fy(idx,:);
    end
else % discrete fourier basis
    fy=fft(x,[],1);
    f=(0:size(fy,1)-1);
    f=min(f,size(fy,1)-f);
    switch(lower(option))
        case 'full'
            idx=find(f<filter(1)*(rt*size(fy,1))|f>=filter(2)*(rt*size(fy,1)));
            %idx=idx(idx>1);
            fy(idx,:)=0;
            k=1; %2*size(fy,1)*(min(.5,filter(2)*rt)-max(0,filter(1)*rt))/max(1,size(fy,1)-numel(idx));
            y=real(ifft(fy,[],1))*k;
        case 'partial'
            idx=find(f>=filter(1)*(rt*size(x,1))&f<filter(2)*(rt*size(x,1)));
            %if ~any(idx==1), idx=[1,idx]; end
            y=fy(idx,:);
    end
end