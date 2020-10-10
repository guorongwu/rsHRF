function [xwiener,sigma] = rsHRF_iterative_wiener_deconv(y,h,Iterations,flag_print)
% Wiener deconvolution estimate
% y = x*h + n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:A.D. Hillery & R.T. Chin, Iterative Wiener filters for image restoration,
% IEEE Trans. Signal Processing, 1991, vol.39, 1892-1899.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<3
    Iterations = 10;
    flag_print = 0;
elseif nargin<4
    flag_print = 0;
end
N = size(y,1);
nh = size(h,1);
if N~= nh
    h = [h; zeros(N-nh,1)];
end
H = fft(h);
Y = fft(y);
[c,l] = wavedec(y,1,'db2');
sigma = wnoisest(c,l,1); 
if flag_print
    fprintf('sigma %2.3f\n',sigma);
end
Phh = abs(H).^2;
sqrdtempnorm = (((norm(y-mean(y))^2  - (N-1)*sigma^2))/(norm(h,1))^2); 
Nf = sigma^2*N;
tempreg = Nf/(sqrdtempnorm);
% using minkowski's ineq
Pxx0 = abs( Y .* (conj(H)) ./(Phh + N*tempreg)).^2;
Pxx = Pxx0;
for i = 1:Iterations
    M = (conj(H) .* Pxx .* Y) ./ (Phh.*Pxx + Nf);% wiener estimate
    PxxY = (Pxx .* Nf) ./ (Phh.*Pxx + Nf);
    Pxx = PxxY + abs(M).^2;
end
WienerFilterEst = (conj(H) .* Pxx) ./ ((abs(H).^2 .* Pxx) + Nf);
xwiener  = real(ifft(WienerFilterEst .* Y));