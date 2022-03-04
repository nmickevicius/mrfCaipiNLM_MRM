# Through-Plane Accelerated MRF Reconstruction

This repository contains the source code referenced in the following work accepted for publication in Magnetic Resonance in Medicine. Please cite this work if you find this code useful in your application. The code that implements the solution to Equation 1 is contained in the 'processMrfData' python script. 

Nikolai J. Mickevicius and Carri K. Glide-Hurst. Low-rank inversion reconstruction for through-plane accelerated radial MR fingerprinting applied to relaxometry at 0.35T. Magnetic Resonance in Medicine (2022).

## Installation

### Installing Required Python Packages
1. Open a terminal and navigate to the root mrfCaipiNLM_MRM repository directory and run the following command. It is recommended to create a virtual environment so the requirements for this package do not overwrite the preferred versions used on your machine locally.
    1. ```pip3 install -r requirements.txt```
    2. The above command will install the required python libraries necessary including a modified version of torchkbnufft that implements temporally low-rank NUFFT operations.

### Compiling Non-Local Means Denoising Code
1. The non-local means (NLM) denoising code was written in C with a matlab mex interface. 
2. In a matlab command window, change the working directory to the root mrfCaipiNLM_MRM repository and run the folloiwng command.
    1. ```mex mcnlmdn_mex.c```

### Running the Reconstruction Code 
1. A script was provided to reconstruct the R=1 and R=2 NIST/ISMRM phantom datasets using the methods described in the paper.
2. In a matlab command window, change the working directory to the root mrfCaipiNLM_MRM repository and run the folloiwng command.
    1. ```run demo_mrfCaipiNLM.m```
3. A figure like the following will show up for each reconstructed slice (i.e., a single slice for R=1 and two slices for R=2)
4. ![Alt text](data/examplePlot.png?raw=true "Title")
5. If you open the 'demo_mrfCaipiNLM.m' file, you can change between R=1 and R=2 datasets, tune reconstruction parameters, and swap between CPU and GPU reconstructions if your setup allows for compuation on CUDA devices. 

