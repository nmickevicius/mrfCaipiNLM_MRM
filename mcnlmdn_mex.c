#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    // input
    double *imgr;
    double *imgi;
    int nhood;
    int winsize;
    double beta;
    
    // output variables
    double *imgdnr;
    double *imgdni;
    
    // dimensions
    int NR, NC, NZ;
    const mwSize *inpdims;
    
    // indices
    long i, j, k, r, c, hr, hc, pr, pc, pz, dnr, dnc, idx, idxmov, cidx, ridx, cidxmov, ridxmov;
    
    int nw = (winsize-1)/2;
    double winsq, w, wsum, diffr, diffi, wmax;
    
    // read the input
    imgr = mxGetPr(prhs[0]);
    imgi = mxGetPi(prhs[0]);
    winsize = (int)(mxGetScalar(prhs[1]));
    nhood = (int)(mxGetScalar(prhs[2]));
    beta = mxGetScalar(prhs[3]);
    
    inpdims = mxGetDimensions(prhs[0]);
    NR = (int)(inpdims[0]);
    NC = (int)(inpdims[1]);
    NZ = (int)(inpdims[2]);
    winsq = (double)(winsize) * (double)(winsize);
    
    // allocate patch
    double ***patchr = (double ***)mxCalloc(winsize, sizeof(double **));
    double ***patchi = (double ***)mxCalloc(winsize, sizeof(double **));
    double ***movr = (double ***)mxCalloc(winsize, sizeof(double **));
    double ***movi = (double ***)mxCalloc(winsize, sizeof(double **));
    for (i=0; i<winsize; i++) {
        patchr[i] = (double **)mxCalloc(winsize, sizeof(double *));
        patchi[i] = (double **)mxCalloc(winsize, sizeof(double *));
        movr[i] = (double **)mxCalloc(winsize, sizeof(double *));
        movi[i] = (double **)mxCalloc(winsize, sizeof(double *));
        for (j=0; j<winsize; j++) {
            patchr[i][j] = (double *)mxCalloc(NZ, sizeof(double));
            patchi[i][j] = (double *)mxCalloc(NZ, sizeof(double));
            movr[i][j] = (double *)mxCalloc(NZ, sizeof(double));
            movi[i][j] = (double *)mxCalloc(NZ, sizeof(double));
        }
    }
    
    // allocate lambda for each point in search neighborhood
    //double *lam = (double *)mxCalloc(nhood*nhood, sizeof(double));
    //long numPatches = 0;
    //long lp;
    
    // create output array
    plhs[0] = mxCreateNumericArray(3, inpdims, mxDOUBLE_CLASS, mxCOMPLEX);
    imgdnr = mxGetPr(plhs[0]);
    imgdni = mxGetPi(plhs[0]);
    
    
    // loop over rows
    for (r=0; r<NR; r++) {
        for (c=0; c<NC; c++) {
            
            if ((r - nw >= 0) && (r + nw < NR) && (c - nw >= 0) && (c + nw < NC)) {
                
                // get current patch
                for (pr=0; pr<winsize; pr++) {
                    ridx = r - nw + pr;
                    for (pc=0; pc<winsize; pc++) {
                        cidx = c - nw + pc;
                        for (pz=0; pz<NZ; pz++) {
                            idx = pz*NR*NC + cidx*NR + ridx;
                            //idx = pz*NR*NC + (c-nw+pc)*NR + (r-nw+pr);
                            patchr[pr][pc][pz] = imgr[idx];
                            patchi[pr][pc][pz] = imgi[idx];
                        }
                    }
                }
                
                // allocate nhood x nhood array containing weights 
                double **weights = (double **)mxCalloc(nhood, sizeof(double *));
                for (hr=0; hr<nhood; hr++) {
                    weights[hr] = (double *)mxCalloc(nhood, sizeof(double *));
                }
                
                // reset the number of patches compared
                //numPatches = 0;
                for (hr=0; hr<nhood; hr++) {
                    for (hc=0; hc<nhood; hc++) {
                        dnr = -nhood/2 + hr;
                        dnc = -nhood/2 + hc;
                        if ((dnr==0) && (dnc==0)) {
                            weights[hr][hc] = 0.0;
                        }
                        else if ((r+dnr-nw >= 0) && (r+dnr+nw < NR) && (c+dnc-nw >= 0) && (c+dnc+nw < NC)) {
                            
                            w = 0.0;
                            for (pr=0; pr<winsize; pr++) {
                                ridx = r + dnr - nw + pr;
                                for (pc=0; pc<winsize; pc++) {
                                    cidx = c + dnc - nw + pc;
                                    for (pz=0; pz<NZ; pz++) {
                                        idx = pz*NR*NC + cidx*NR + ridx;
                                        movr[pr][pc][pz] = imgr[idx];
                                        movi[pr][pc][pz] = imgi[idx];
                                        
                                        diffr = movr[pr][pc][pz] - patchr[pr][pc][pz];
                                        diffi = movi[pr][pc][pz] - patchi[pr][pc][pz];
                                        w += sqrt(diffr*diffr + diffi*diffi);
                                    }
                                }
                            }
                            w = exp(-beta*w);
                            weights[hr][hc] = w;
                            //lam[numPatches] = w;
                            //numPatches++;
                        }
                    }
                }
                
                // get sum of weights and calculate the maximum value
                wsum = 0.0; 
                wmax = 0.0;
                for (hr=0; hr<nhood; hr++) {
                    for (hc=0; hc<nhood; hc++) { 
                        if (weights[hr][hc] > wmax) {
                            wmax = weights[hr][hc];
                        }
                        wsum += weights[hr][hc];
                    }
                }
                wsum += wmax;
                
                // reconstruct current pixel
                for (hr=0; hr<nhood; hr++) {
                    for (hc=0; hc<nhood; hc++) {
                        dnr = -nhood/2 + hr;
                        dnc = -nhood/2 + hc;
                        if ((r+dnr-nw >= 0) && (r+dnr+nw < NR) && (c+dnc-nw >= 0) && (c+dnc+nw < NC)) {
                            if ((dnr==0) && (dnc==0)) {
                                w = wmax;
                            }
                            else {
                                w = weights[hr][hc];
                            }
                            for (pz=0; pz<NZ; pz++) {
                                idx = pz*NR*NC + c*NR + r;
                                idxmov = pz*NR*NC + (c+dnc)*NR + r + dnr;
                                imgdnr[idx] += imgr[idxmov]*w/wsum;
                                imgdni[idx] += imgi[idxmov]*w/wsum;
                            }
                        }
                    }
                }
                
                /*
                // get maximum value of lambda to be used for (r,c) patch weighting
                // and calculate sum of all lambda values
                wsum = 0.0;
                wmax = 0.0;
                for (lp=0; lp<numPatches; lp++) {
                    if (lam[lp] > wmax) {
                        wmax = lam[lp];
                    }
                    wsum += lam[lp];
                }
                wsum += wmax;
                
                for (hr=0; hr<nhood; hr++) {
                    for (hc=0; hc<nhood; hc++) {
                        dnr = -nhood/2 + hr;
                        dnc = -nhood/2 + hc;
                        
                        if ((r+dnr-nw >= 0) && (r+dnr+nw < NR) && (c+dnc-nw >= 0) && (c+dnc+nw < NC)) {
                            
                            if ((dnr==0) && (dnc==0)) {
                                w = wmax;
                            }
                            else {
                                
                                w = 0.0;
                                for (pr=0; pr<winsize; pr++) {
                                    ridx = r + dnr - nw + pr;
                                    for (pc=0; pc<winsize; pc++) {
                                        cidx = c + dnc - nw + pc;
                                        for (pz=0; pz<NZ; pz++) {
                                            idx = pz*NR*NC + cidx*NR + ridx;
                                            movr[pr][pc][pz] = imgr[idx];
                                            movi[pr][pc][pz] = imgi[idx];
                                            
                                            diffr = movr[pr][pc][pz] - patchr[pr][pc][pz];
                                            diffi = movi[pr][pc][pz] - patchi[pr][pc][pz];
                                            w += sqrt(diffr*diffr + diffi*diffi);
                                        }
                                    }
                                }
                                w = exp(-beta*w/winsq);
                            }
                            
                            for (pz=0; pz<NZ; pz++) {
                                idx = pz*NR*NC + c*NR + r;
                                idxmov = pz*NR*NC + (c+dnc)*NR + r + dnr;
                                imgdnr[idx] += imgr[idxmov]*w/wsum;
                                imgdni[idx] += imgi[idxmov]*w/wsum;
                            }
                            
                        }
                    }
                }
                */
                
                // free weights for each target pixel 
                mxFree(weights);
                
            }
            
        } // c
    } // r
    
    // free working memory
    mxFree(patchr);
    mxFree(patchi);
    mxFree(movr);
    mxFree(movi);
    //mxFree(lam);
    
} // end mexFunction





