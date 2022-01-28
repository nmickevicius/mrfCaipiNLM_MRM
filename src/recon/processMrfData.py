import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchkbnufft as tkbn
import torch.optim
from torch.autograd import Variable
import scipy.ndimage as ndimage
import numpy as np
import copy
import scipy.io as sio 
import sigpy as sp
import sigpy.mri as mr
import os
from datetime import datetime
import h5py 
import matplotlib.pyplot as plt
import sys

class LowRankNufftOperator(torch.nn.Module):
    def __init__(self, ktraj, im_size, grid_size, U, ksize=6, rdcf=None, sens=None, phi=None, l2lam=None, device=None):
        super(LowRankNufftOperator,self).__init__()
        # ktraj   - k-space trajectory [nsegs, T, nread, 2] scaled [-pi,pi] torch.float32
        # im_size - tuple with (rows,cols) 
        # U       - low-dimensional subspace [nrf, K] torch.complex64
        # sens    - sensitivity maps [batch, slices, coils, 1, nx, nx] torch.complex64
        # rdcf    - square root of density compensation [nsegs, T, nread] torch.float32
        # phi     - CAIPI phase modulation over slices [slices, nsegs, T, nread] torch.complex64

        # get some dimensions 
        self.nseg, self.T, self.nread, _ = ktraj.shape 
        self.K = U.shape[1]
        self.im_size = im_size 
        self.nrows, self.ncols = im_size 
        self.grid_size = grid_size 
        self.ksize = ksize 
        self.l2lam = l2lam
        self.device = device

        # format k-space trajectory
        kx = ktraj[...,0].reshape(-1,)
        ky = ktraj[...,1].reshape(-1,)
        self.ktraj = torch.stack([kx,ky],axis=0)

        # tile and reshape U
        self.U = U.unsqueeze(1).unsqueeze(0)                    # [1, T, 1, K]
        self.U = torch.tile(self.U, (self.nseg,1,self.nread,1)) # [nseg, T, nread, K]
        self.U = self.U.permute(3,0,1,2).reshape(self.K,-1)     # [K, nseg*T*nread]
        self.U = self.U.unsqueeze(0).permute(2,0,1)             # [nseg*T*nread,1,K]

        # calculate kaiser-bessel NUFFT operator in order to get apodization kernel
        F = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, numpoints=self.ksize)
        self.apod = F.scaling_coef[None,None,None,None,...]
        self.apod = self.apod.to(self.device)

        # get real/imag sparse gridding matrices from a custom function I added 
        # to torchkbnufft to build-in the subspace constraints directly into the 
        # sparse gridding matrix
        self.GU = tkbn.calc_lowrank_spmatrix(self.ktraj, self.U, self.im_size, self.grid_size, numpoints=self.ksize)

        # we can check here to see if real(GU) or imag(GU) contains anything but zeros
        self.real_interp = False 
        self.imag_interp = False 
        if torch.sparse.sum(self.GU[0]) == 0.0:
            print('Interpolation matrices are all imaginary')
            self.imag_interp = True 
        elif torch.sparse.sum(self.GU[1]) == 0.0:
            print('Interpolation matrices are all real')
            self.real_interp = True 
        
        # pre-calculate conjugate transpose of GU for speed
        self.GUh = (self.GU[0].transpose(1,0), -1.0*self.GU[1].transpose(1,0))

        # reshape density compensation and phase modulation
        if rdcf is not None:
            self.rdcf = rdcf.reshape(-1,).to(torch.complex64)
        else:
            self.rdcf = None
        if phi is not None:
            self.nslc = phi.shape[0]
            self.phi = phi.reshape(self.nslc,-1)
        else:
            self.phi = None
            self.nslc = 1

        # store coil maps 
        self.sens = sens 
        if self.sens is not None:
            self.ncoils = self.sens.shape[2]
        else:
            self.ncoils = 1

    def forward(self, x):
        # x - [batch, nslc, 1, K, nrows, ncols]
        batch = x.shape[0]
        Ax = self.apod * x                                     # scaling by IFT of KB interpolation kernel
        if self.sens is not None:
            Ax = Ax * self.sens                                # multiply by coil sensitivity maps 
        Ax = torch.fft.fft2(Ax, s=self.grid_size, dim=(-2,-1)) # Fourier transform to k-space [batch, nslc, coils, K, nrows, ncols]
        Ax = Ax.reshape(batch*self.nslc*self.ncoils,-1)        # reshape to [batch*nslc*ncoils, K*nrows*ncols]
        Ax = Ax.permute(1,0)                                   # transpose to [K*nrows*ncols, batch*nslc*ncoils]

        if self.real_interp:
            GUAxr = torch.sparse.mm(self.GU[0],Ax.real)
            GUAxi = torch.sparse.mm(self.GU[0],Ax.imag)
        elif self.imag_interp:
            GUAxr = -1.0*torch.sparse.mm(self.GU[1],Ax.imag)
            GUAxi = torch.sparse.mm(self.GU[1],Ax.real)
        else:
            GUAxr = torch.sparse.mm(self.GU[0],Ax.real) - torch.sparse.mm(self.GU[1],Ax.imag) # real part of de-gridded k-space [segs*T*nread, batch*nslc*ncoils]
            GUAxi = torch.sparse.mm(self.GU[0],Ax.imag) + torch.sparse.mm(self.GU[1],Ax.real) # imag part of de-gridded k-space [segs*T*nread, batch*nslc*ncoils]
        Ax = torch.complex(GUAxr, GUAxi).reshape(-1,batch,self.nslc,self.ncoils)          # complexify and reshape to [segs*T*nread, batch, nslc, ncoils]

        # phase modulation and slice summation operator 
        y = Ax.clone()
        for r in range(self.nslc):
            for s in range(self.nslc):
                if s is not r:
                    ys = Ax[:,:,s,:].clone() # [segs*T*nread, batch, ncoils]
                    ys = ys * self.phi[s,:].unsqueeze(-1).unsqueeze(-1)
                    ys = ys * torch.conj(self.phi[r,:].unsqueeze(-1).unsqueeze(-1))
                    y[:,:,r,:] += ys 

        # density compensation 
        if self.rdcf is not None:
            y *= self.rdcf[:,None,None,None]

        return y.permute(1,2,0,3) # return with shape [batch, nslc, segs*T*nread, ncoils]
        
    def adjoint(self, y, senseCoilComb=True):
        # y [ batch, nslc, segs*T*nread, ncoils]
        batch = y.shape[0]
        ncoils = y.shape[-1]
        y = y.permute(2,0,1,3) # [segs*T*nread, batch, nslc, ncoils]

        # density compensation 
        if self.rdcf is not None:
            yc = y * self.rdcf[:,None,None,None]
        else:
            yc = y.clone()

        # gridding 
        yc = yc.reshape(y.shape[0],-1) # [segs*T*nread, batch*nslc*ncoils]
        if self.real_interp:
            Ahy = torch.complex(torch.sparse.mm(self.GUh[0],yc.real), torch.sparse.mm(self.GUh[0],yc.imag)) # [K*prod(grid_size), batch*nslc*ncoils]
        elif self.imag_interp:
            Ahy = torch.complex(torch.sparse.mm(-1.0*self.GUh[1],yc.imag), torch.sparse.mm(self.GUh[1],yc.real)) # [K*prod(grid_size), batch*nslc*ncoils]
        else:
            Ahy = torch.complex(torch.sparse.mm(self.GUh[0],yc.real) - torch.sparse.mm(self.GUh[1],yc.imag),
                            torch.sparse.mm(self.GUh[0],yc.imag) + torch.sparse.mm(self.GUh[1],yc.real)) # [K*prod(grid_size), batch*nslc*ncoils]

        Ahy = Ahy.reshape(self.K, self.grid_size[0], self.grid_size[1], batch, self.nslc, ncoils)
        Ahy = Ahy.permute(3,4,5,0,1,2) # [batch, nslc, ncoils, K, nx, nx]
        Ahy = torch.fft.ifft2(Ahy, s=self.grid_size, dim=(-2,-1))
        Ahy = Ahy[...,0:self.im_size[0],0:self.im_size[1]] # crop 
        if self.sens is not None:
            Ahy = torch.sum(Ahy*torch.conj(self.sens), dim=2, keepdim=True)
        Ahy *= torch.conj(self.apod)

        return Ahy 

    def normal(self,x):
        xhat = self.adjoint(self.forward(x))
        if self.l2lam:
            xhat = xhat + self.l2lam * x 
        return xhat

    def prepraw(self,y):
        # y - [batch, nseg*T*nread, coils]
        if self.rdcf is not None:
            yc = y * self.rdcf[None,:,None]
        else:
            yc = y.clone()
        yc = yc.unsqueeze(1) 
        if self.phi is not None:
            yc = yc * self.phi[None,:,:,None]
        return yc 

def conjgrad(A, b, x, niter, l2lam):
    '''
    This is a modified version of the conjgrad() function in the deepinpy
    library (https://github.com/utcsilab/deepinpy)
    '''
    # Solve (A'*A)*x = b for x, where b = A'*y

    def dot(x1,x2):
        return torch.sum(x1.real*x2.real + x1.real*x2.real, dim=list(range(1, len(x1.shape))))
    
    # explicitly remove r from the computational graph
    r = b.new_zeros(b.shape, requires_grad=False)

    if l2lam is not None:
        r = b - A.normal(x) + l2lam*x
    else:
        r = b - A.normal(x)

    p = r 
    rsnot = dot(r,r)
    rsold = rsnot 
    rsnew = rsnot 
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    for i in range(niter):
        print(i)
        if l2lam is not None:
            Ap = A.normal(p) + l2lam*p
        else:
            Ap = A.normal(p)
        pAp = dot(p,Ap)
        alpha = (rsold / pAp).reshape(reshape)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r,r)
        beta = (rsnew / rsold).reshape(reshape)
        rsold = rsnew
        p = beta * p + r
        
    return x

class CG(torch.autograd.Function):
    # autograd function for conjgrad()
    def __init__(self, A, niter, l2lam):
        self.A = A 
        self.niter = niter 
        self.l2lam = l2lam

    def forward(self, b, x0):
        x = conjgrad(self.A, b, x0, self.niter, self.l2lam)
        return x 

    def backward(self, dx):
        db = conjgrad(self.A, dx, None, self.niter, self.l2lam)
        return db

# locally low rank soft threshold. Performs batch SVD + soft-thresholding of singular values
def llr_soft_thresh(lam, block_op, W, S, input):
    '''
    Obtained from https://github.com/jtamir/t2sh-python/blob/master/T2%20Shuffling%20Sigpy%20Demo.ipynb
    '''
    input_reshape = input.reshape((input.shape[0], input.shape[1], -1))
    data = block_op * input_reshape
    data_reshape = data.reshape((data.shape[0], data.shape[1], -1, W*W))
    u, s, vh = np.linalg.svd(data_reshape, full_matrices=False)
    s_st = sp.soft_thresh(lam, s)
    data_reshape_st = u * s_st[..., None, :] @ vh
    output = block_op.H * data_reshape_st.reshape(data.shape)
    return output.reshape((input.shape))

def admm_recon_llr(y, A, admmiters_llr, cgiters_llr, admm_rho, lambda_llr, W):

    def cg_admm(x, fn, rhs, niter):
        def dot(x1,x2):
            return torch.sum(x1.real*x2.real + x1.real*x2.real, dim=list(range(1, len(x1.shape))))
        r = rhs - fn(x)
        r.requires_grad = False 
        p = r 
        rsnot = dot(r,r)
        rsold = rsnot 
        rsnew = rsnot 
        reshape = (-1,) + (1,) * (len(x.shape) - 1)
        for i in range(niter):
            Ap = fn(p)
            pAp = dot(p,Ap)
            alpha = (rsold / pAp).reshape(reshape)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = dot(r,r)
            beta = (rsnew / rsold).reshape(reshape)
            rsold = rsnew
            p = beta * p + r
        return x

    x = A.adjoint(y) # [batch, Rkz, 1, K, nx, nx]
    x.requires_grad = False
    assert x.shape[0] == 1, 'Batch sizes > 1 not supported'
    rhs = x.clone()
    K = x.shape[3]
    nx = x.shape[-1]

    z = torch.zeros(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    # z_old = torch.zeros(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    u = torch.zeros(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)

    A.l2lam = None

    block_op = sp.linop.ArrayToBlocks([nx, nx, K], [W, W, 1], [W, W, 1])
    
    for a in range(admmiters_llr):

        #print('ADMM iteration %i/%i'%(a+1,admmiters_llr))

        # update x 
        fn = lambda a: admm_rho*a + A.normal(a)
        b = rhs + admm_rho * (z - u)
        x = cg_admm(x, fn, b, cgiters_llr)

        # update z
        xpu = x + u
        if admm_rho > 0:
            for s in range(x.shape[1]):
                xpus = xpu[0,s,0,:,:,:].permute(1,2,0).detach().cpu().numpy() # [nx,nx,K]
                znp = llr_soft_thresh(lambda_llr/admm_rho, block_op, W, W, xpus).transpose(2,0,1) # [K, nx, nx]
                z[0,s,0,:,:,:] = torch.tensor(znp, dtype=z.dtype, device=z.device)

        # update u
        u = xpu - z

    return x

def espiritCalib(cimgs):
    F = sp.linop.FFT((cimgs.shape), axes=(1,2), center=True)
    cref = F * cimgs
    csm = mr.app.EspiritCalib(cref,crop=0.95,show_pbar=False).run()
    return csm



# input args 
kspFile = sys.argv[1]
trajFile = sys.argv[2]
basisFile = sys.argv[3]
admmIters = int(sys.argv[4])
cgIters = int(sys.argv[5])
admmRho = float(sys.argv[6])
llrLambda = float(sys.argv[7])
llrWin = int(sys.argv[8])
gpuFlag = int(sys.argv[9])
outFile = sys.argv[10]

# determine computational device 
if gpuFlag:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# load MRF data 
dfmrf = sio.loadmat(kspFile)
data = dfmrf['data']
phi = dfmrf['phi']
nxos, ncoils, nrf, nsegs = data.shape 
nx = int(nxos/2)

# load trajectory 
dftraj = sio.loadmat(trajFile)
traj = dftraj['traj']
dcf = dftraj['dcf']

# load subspace basis functions
dfbasis = sio.loadmat(basisFile)
basis = dfbasis['basis']
K = basis.shape[1]

# convert data to torch tensor
y = data.transpose(1,3,2,0)
y = y.reshape(y.shape[0],-1) # coils, nsegs*nrf*nread
y = torch.tensor(y, dtype=torch.complex64, device=device)

# convert trajectory to torch tensor 
ktraj = torch.tensor(traj, dtype=torch.float32, device=device)
rdcf = torch.tensor(np.sqrt(dcf), dtype=torch.complex64, device=device).unsqueeze(0).unsqueeze(0)
rdcf = torch.tile(rdcf,(nsegs,nrf,1))

# convert basis functions to torch tensor
U = torch.tensor(basis, dtype=torch.complex64, device=device) 

# convert phase modulation to torch tensor 
phi_torch = torch.tensor(phi, dtype=torch.complex64, device=device)

# make low-rank nufft operator 
im_size = (nx,nx)
grid_size = (nxos,nxos) 
A = LowRankNufftOperator(ktraj, im_size, grid_size, U, ksize=4, rdcf=rdcf, sens=None, phi=phi_torch, l2lam=None, device=device)

# get data in proper format 
yinp = y.permute(1,0).unsqueeze(0)
yinp = A.prepraw(yinp)

# get multi-coil images 
xhat = A.adjoint(yinp)

# calculate coil sensitivity maps 
csm = []
for sms in range(xhat.shape[1]):
    ref = xhat[0,sms,:,0,:,:].detach().cpu().numpy()
    csm.append(espiritCalib(ref))
csm = np.stack(csm,axis=0)
sens = torch.tensor(csm, dtype=torch.complex64, device=device).unsqueeze(2).unsqueeze(0)

# update imaging model 
A.sens = sens
A.ncoils = ncoils

# get adjoint of data including coil combination
xhat = A.adjoint(yinp)

# # do the linear reconstruction 
# with torch.no_grad():
#     cgfn = CG(A, cgIters, None)
#     xlin = cgfn.forward(xhat.clone(), xhat.clone())

# do a locally-low rank regularized reconstruction
xlin = admm_recon_llr(yinp, A, admmIters, cgIters, admmRho, llrLambda, llrWin)

# save the reconstruction to output file 
recon = xlin.detach().cpu().numpy()[0,:,0,:,:,:] # [sms, K, nrows, ncols]
recon = np.transpose(recon,(2,3,1,0))
sio.savemat(outFile,{'recon':recon})






