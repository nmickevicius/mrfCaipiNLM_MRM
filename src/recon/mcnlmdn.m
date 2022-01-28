function imgdn = mcnlmdn(img, win, nhood, beta)
%
% Non-local means denoising of a multi-contrast 2D image. The function is
% parameterized by the inputs below. 
%
% INPUT:
%   img - noisy input image [nrows, ncolumns, ncontrasts] (complex double)
%   win - sliding window size (uniform in both the row and col dimensions)
%   nhood - search neighborhood size 
%   beta - denoising value (smaller = less noise) 

% make sure input is complex 
inputIsReal = false;
if isreal(img)
    inputIsReal = true;
    img = complex(img, zeros(size(img)));
end

assert(length(win) == 1, 'win should be a scalar');
assert(mod(win,2) == 1, 'win should be odd');
assert(mod(nhood,2) == 0, 'nhood should be even');
assert(beta > 0, 'beta should be positive');

% make sure input is double 
img = double(img);
win = double(win);
nhood = double(nhood);
beta = double(beta);

% call the mex function that performs the denoising
imgdn = mcnlmdn_mex(img, win, nhood, beta);

% if input was real-valued, return the real part of the denoised image
if inputIsReal
    imgdn = real(imgdn);
end







