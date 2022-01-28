%% Set-Up

addpath(genpath('src'));

%% k-Space Data File 

% acceleration factor (must be 1 or 2)
R = 1; 

if R == 1
    kspFile = fullfile('data','data_nist_ismrm_phantom_slc23_R1.mat');
elseif R == 2
    kspFile = fullfile('data','data_nist_ismrm_phantom_slc3_R2.mat'); 
else
    error('Acceleration factor must be 1 or 2');
end

%% Load k-Space Trajectory

trajFile = fullfile('data','data_nist_ismrm_phantom_traj.mat');

%% Subspace Basis Function File 

basisFile = fullfile('data','basis.mat');

%% Dictionary File 

dictFile = fullfile('data','dictionary.mat');

%% Reconstruction Parameters 

% flag to use GPU for CUDA devices compatible with installed PyTorch version
useGpuFlag = 0; % set to 0 for CPU and 1 for GPU

if R == 1
    
    % set the parameters for a linear reconstruction for R=1
    numIterationsADMM = 1; % number of ADMM iterations
    numIterationsCG = 20;  % number of CG iterations for linear reconstruction
    admmRho = 0;           % rho must be 0.0 for linear reconstruction
    llrLambda = 0;         % lambda must be 0.0 for linear reconstruction
    llrWin = 8;
    
elseif R == 2
    
    % set the parameters for ADMM with LLR regularization for R=2
    numIterationsADMM = 8;
    numIterationsCG = 5;  % number of CG iterations for linear reconstruction
    admmRho = 0.001;
    llrLambda = 0.0001;
    llrWin = 8;
    
end

% get time-stamped output file name 
nowstr = datestr(now);
nowstr(nowstr == ' ') = '_';
outFile = sprintf('%s_RECON_%s.mat',kspFile(1:(end-4)), nowstr);

%% Run the Reconstruction in Python

ts = tic;
comm = sprintf('python3 src/recon/processMrfData.py %s %s %s %i %i %f %f %i %i %s', kspFile, trajFile, basisFile, numIterationsADMM, numIterationsCG, admmRho, llrLambda, llrWin, useGpuFlag, outFile);
system(comm);
tt = toc(ts);
fprintf('Reconstruction time took %f seconds\n', tt);

%% Do Non-Local Means Denoising

load(outFile,'recon');
scl = max(abs(recon(:)));

beta = 32;
win = 3;      % window size 
nhood = 64;   % neighborhood search size

clear recon_nlm;
recon_nlm = zeros(size(recon));
recon_lpf = zeros(size(recon));
for slc = 1:size(recon,4)
    img = double(recon(:,:,:,slc)./scl);
    recon_nlm(:,:,:,slc) = mcnlmdn(img, win, nhood, beta); 
    for k = 1:size(img,3)
        recon_lpf(:,:,k,slc) = conv2(img(:,:,k),ones(2,2)./4,'same'); % low-pass filter for comparison
    end
end
recon_nlm = scl.*single(recon_nlm);
recon_lpf = single(recon_lpf);


%% Dictionary Matching 

load(basisFile,'basis');
load(dictFile,'evol','lut');
elr = basis'*evol; clear evol;
lut = [lut, ones(size(lut,1),1)];
nq = size(lut,2);
qmaps = zeros(size(recon_nlm,1), size(recon_nlm,2), nq+1, size(recon_nlm,4));
qmaps_nlm = zeros(size(recon_nlm,1), size(recon_nlm,2), nq+1, size(recon_nlm,4));
qmaps_lpf = zeros(size(recon_nlm,1), size(recon_nlm,2), nq+1, size(recon_nlm,4));
b1 = ones(size(recon,1),size(recon,2));
for slc = 1:size(recon,4)
    qmaps(:,:,:,slc) = dictionaryMatchExtB1(recon(:,:,:,slc), b1, elr, lut, 1000);
    qmaps_nlm(:,:,:,slc) = dictionaryMatchExtB1(recon_nlm(:,:,:,slc), b1, elr, lut, 1000);
    qmaps_lpf(:,:,:,slc) = dictionaryMatchExtB1(recon_lpf(:,:,:,slc), b1, elr, lut, 1000);
end


%% Save Final Output to Disk

finalFile = sprintf('%s_OUTPUT_v1_%s.mat',kspFile(1:(end-4)), nowstr);
i = 2;
while exist(finalFile,'file')
    finalFile = sprintf('%s_OUTPUT_v%i_%s.mat',kspFile(1:(end-4)), i, nowstr);
    i = i + 1;
end
save(finalFile, 'recon', 'recon_nlm', 'recon_lpf', 'qmaps', 'qmaps_nlm', 'qmaps_lpf');


%% Show Results

load(finalFile);

% colormap limits 
lims = [0,3; 0,2];

a0 = recon_nlm(:,:,1,:);
msk = abs(a0) > 0.04*max(abs(a0(:)));
msk = imfill(msk,ones(3,3),'holes');
msk = bwareaopen(msk,100,ones(3,3));

sph = 0.05; 
spv = 0.05;

for slc = 1:size(qmaps,4)
    figure; 
    set(gcf,'color','w');
    for q = 1:2
        subaxis(2,3,(q-1)*3+1,'spacingvert',spv,'spacinghoriz',sph);
        dispmrf(qmaps(:,:,q,slc).*msk(:,:,1,slc),lims(q,:),'parula');
        set(gca,'FontSize',24);
        if q == 1
            title('No Denoising','Interpreter','Latex');
            ylabel('$T_1$ [s]','Interpreter','Latex');
        elseif q == 2
            ylabel('$T_2$ [s]','Interpreter','Latex');
        end
        subaxis(2,3,(q-1)*3+2,'spacingvert',spv,'spacinghoriz',sph);
        dispmrf(qmaps_lpf(:,:,q,slc).*msk(:,:,1,slc),lims(q,:),'parula');
        set(gca,'FontSize',24);
        if q == 1
            title('Low-Pass Filtered','Interpreter','Latex');
        end
        subaxis(2,3,(q-1)*3+3,'spacingvert',spv,'spacinghoriz',sph);
        dispmrf(qmaps_nlm(:,:,q,slc).*msk(:,:,1,slc),lims(q,:),'parula');
        set(gca,'FontSize',24);
        if q == 1
            title('NLM Denoised','Interpreter','Latex');
        end
    end
end

