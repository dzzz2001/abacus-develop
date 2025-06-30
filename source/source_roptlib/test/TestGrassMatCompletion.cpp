#include "test/TestGrassMatCompletion.h"

using namespace ROPTLIB;

void testGrassMatCompletion(void)
{
//    genrandseed(1);
    //for (integer i = 0; i < 1; i++)
    //{
//    integer d = 50, N = 1000, r = 5;
    integer d = 10, N = 50, r = 2;
//    integer d = 5, N = 10, r = 2;
    
    Grassmann Domain(d, r);
    Domain.SetHasHHR(true);
    Domain.SetIsIntrApproach(true);
    Variable GrassX = Domain.RandominManifold();
    
    // Generate the matrices in the matrix completion approximation problem.
    integer dim = (d + N - r) * r;
    realdp OS = 3;
    integer nz = (OS * dim > d * N) ? d * N : OS * dim;
    integer *ir = new integer[nz * 2];
    integer *jc = ir + nz;
    
    integer *tmpforidx = new integer[d * N];
    for (integer i = 0; i < d * N; i++)
        tmpforidx[i] = i;
    /*nz number of indices*/
    integer idx = 0, itmp;
    for (integer i = 0; i < nz; i++)
    {
        /*idx is an integer in [0, m - i - 1]*/
        idx = static_cast<integer> ((d * N - i) * genrandreal());
        while (idx >= d * N - i)
            idx = static_cast<integer> ((d * N - i) * genrandreal());
        /*the chosen idx is put at the end of the array*/
        itmp = tmpforidx[d * N - i - 1];
        tmpforidx[d * N - i - 1] = tmpforidx[idx];
        tmpforidx[idx] = itmp;
    }
    for (integer i = 0; i < nz; i++)
    {
        /*tmpforidx[nz - 1 - i]*/
        ir[i] = static_cast<integer> (tmpforidx[nz - 1 - i] / N);
        jc[i] = tmpforidx[nz - 1 - i] - N * ir[i];
    }
    delete[] tmpforidx;
    
    integer dr = d * r, nr = N * r;
    realdp *A_U = new realdp[dr];
    realdp *A_V = new realdp[nr];
    for (integer i = 0; i < d * r; i++)
    {
        A_U[i] = genrandnormal();
    }
    for (integer i = 0; i < N * r; i++)
    {
        A_V[i] = genrandnormal();
    }
    
    realdp *V = new realdp[nz];
    for (integer i = 0; i < nz; i++)
    {
        V[i] = 0;
        for (integer j = 0; j < r; j++)
        {
            V[i] += A_U[ir[i] + j * d] * A_V[jc[i] + j * N];
        }
    }
    delete[]A_U;
    delete[]A_V;
    GrassMatCompletion Prob(ir, jc, V, nz, N, d, r);
    Prob.SetDomain(&Domain);
    
    RBFGS *RBFGSsolver = new RBFGS(&Prob, &GrassX);
    
//    RBFGSsolver->Stop_Criterion = PSSUBGRAD;
    RBFGSsolver->CheckParams();
    RBFGSsolver->Run();
    delete RBFGSsolver;
    
//    RAMSGRADSP *RAMSGRADsolver = new RAMSGRADSP(&Prob, &GrassX);
//    RAMSGRADsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
//    RAMSGRADsolver->Max_Iteration = 200;
//    RAMSGRADsolver->BatchSize = 10;
//    RAMSGRADsolver->OutputGap = 10;
//    RAMSGRADsolver->Tolerance = 1e-6;
//    //RAMSGRADsolver->LMrestart = true;
//    RAMSGRADsolver->CheckParams();
//    RAMSGRADsolver->Run();
//    delete RAMSGRADsolver;
//    
//    RADAMSP *RADAMsolver = new RADAMSP(&Prob, &GrassX);
//    RADAMsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
//    RADAMsolver->Max_Iteration = 200;
//    RADAMsolver->BatchSize = 10;
//    RADAMsolver->OutputGap = 10;
//    RADAMsolver->Tolerance = 1e-6;
//    //RADAMsolver->LMrestart = true;
//    RADAMsolver->CheckParams();
//    RADAMsolver->Run();
//    delete RADAMsolver;
//    
//    RSGD *RSGDsolver = new RSGD(&Prob, &GrassX);
//    RSGDsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
//    RSGDsolver->Max_Iteration = 200;
//    RSGDsolver->BatchSize = 10;
//    RSGDsolver->OutputGap = 10;
//    RSGDsolver->Tolerance = 1e-6;
////    RSGDsolver->LMrestart = true;
//    RSGDsolver->CheckParams();
//    RSGDsolver->Run();
//    delete RSGDsolver;
    
    
    for (integer i = 1; i <= 1; i++)
    {
        realdp lrate = 0.005*i;
        integer BatchSize = 10;
//        RSVRG *RSVRGsolver = new RSVRG(&Prob, &GrassX);
////        RSVRGsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
////        RSVRGsolver->Max_Iteration = 10;
////        RSVRGsolver->OutputGap = 10;
////        RSVRGsolver->Tolerance = 1e-6;
////        RSVRGsolver->Initstepsize = lrate;
////        RSVRGsolver->FreqM = 5*N;
////        RSVRGsolver->BatchSize = BatchSize;
//        //LRBFGSsolver->LMrestart = true;
//        RSVRGsolver->CheckParams();
//        RSVRGsolver->Run();
//        delete RSVRGsolver;
//        
//        SVRLRBroydenFamily *SVRLRBroydensolver = new SVRLRBroydenFamily(&Prob, &GrassX);
////        SVRLRBroydensolver->Verbose = FINALRESULT;
////        SVRLRBroydensolver->Max_Iteration = 10;
////        SVRLRBroydensolver->OutputGap = 10;
////        SVRLRBroydensolver->LengthSY = 6;
////        SVRLRBroydensolver->Tolerance = 1e-6;
////        SVRLRBroydensolver->Initstepsize = lrate;
////        SVRLRBroydensolver->FreqM = 5*N;
////        SVRLRBroydensolver->BatchSize = BatchSize;
////        //LRBFGSsolver->LMrestart = true;
//        SVRLRBroydensolver->CheckParams();
//        SVRLRBroydensolver->Run();
//        delete SVRLRBroydensolver;
//        
//        SVRLRBFGS *SVRLRBFGSsolver = new SVRLRBFGS(&Prob, &GrassX);
//        SVRLRBFGSsolver->Verbose = FINALRESULT;
//        SVRLRBFGSsolver->Max_Iteration = 10;
//        SVRLRBFGSsolver->OutputGap = 10;
//        SVRLRBFGSsolver->LengthSY = 6;
//        SVRLRBFGSsolver->Tolerance = 1e-6;
//        SVRLRBFGSsolver->Initstepsize = lrate;
//        SVRLRBFGSsolver->FreqM = 5*N;
//        SVRLRBFGSsolver->BatchSize = BatchSize;
//        //LRBFGSsolver->LMrestart = true;
//        //LRBFGSsolver->CheckParams();
//        SVRLRBFGSsolver->Run();
//        delete SVRLRBFGSsolver;
    }

    delete[] V;
    delete[] ir;
};

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 4)
    {
        mexErrMsgTxt("The number of arguments should be at least four.\n");
    }
    
    realdp *A, *X;
    A = mxGetPr(prhs[0]);
    X = mxGetPr(prhs[1]);
    /* dimensions of input matrices */
    integer d, n, r, HasHHR, nzmax, IsIntr;
    size_t *ir, *jc;
    nzmax = mxGetNzmax(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    d = mxGetM(prhs[0]);
    r = mxGetN(prhs[1]);
    n = mxGetN(prhs[0]);
    
    /*Check the correctness of the inputs*/
    if(mxGetM(prhs[1]) != d )
    {
        mexErrMsgTxt("The size of the initial X is not correct!\n");
    }
    
    HasHHR = static_cast<integer> (mxGetScalar(prhs[2]));
    
    genrandseed(0);
    
    CheckMemoryDeleted = new std::map<integer *, integer>;
    //    testStieBrockett(B, D, n, p, X, Xopt);
    
    Grassmann Domain(d,r);
    Variable initX = Domain.RandominManifold();
    realdp *initXptr = initX.ObtainWriteEntireData();
    for(integer i = 0; i < d * r; i++)
        initXptr[i] = X[i];
    
    integer *inir = new integer[2 * nzmax];
    integer *injc = inir + nzmax;
    
    for (integer i = 0; i < n; i++)
    {
        for (unsigned long long j = jc[i]; j < jc[i + 1]; j++)
        {
            /*row: ir[j], column: i, entry: A[j]*/
            inir[j] = ir[j];
            injc[j] = i;
        }
    }
    
    GrassMatCompletion Prob(inir, injc, A, nzmax, n, d, r);
    Prob.SetDomain(&Domain);
    
    Domain.SetHasHHR(HasHHR != 0);
    
//    IsIntr = static_cast<integer> (mxGetScalar(prhs[4]));
//    Domain.SetIsIntrApproach(IsIntr);
    //Domain.SetIsIntrApproach(false);
    //Domain.CheckParams();
    
    // Call the function defined in DriverMexProb.h
    ParseSolverParamsAndOptimizing(prhs[3], &Prob, &initX, plhs);
    
    delete[] inir;
    
    std::map<integer *, integer>::iterator iter = CheckMemoryDeleted->begin();
    for (iter = CheckMemoryDeleted->begin(); iter != CheckMemoryDeleted->end(); iter++)
    {
        if (iter->second != 1)
            printf("Global address: %p, sharedtimes: %d\n", iter->first, iter->second);
    }
    delete CheckMemoryDeleted;
    return;
}

#endif
