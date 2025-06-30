
#include "test/TestGrassPCA.h"

using namespace ROPTLIB;

void testGrassPCA(void)
{
    //unsigned tt = (unsigned)time(NULL);
    //tt = 2; /*The following test is only for random seed zero*/
    //std::cout << "seed SB:" << tt << std::endl;//---
    //genrandseed(tt);
    // size of the domain
    integer n = 3;
    integer p = 2;
    integer N = 50;
    Vector A(n, N);
    A.RandGaussian();
    //A.RandUnform();
    // Obtain an initial iterate
    Grassmann Domain(n, p);
    Variable GrassX = Domain.RandominManifold();
    // Define the problem
    GrassPCA Prob(A, N, n, p);
    Domain.SetHasHHR(true);
    Prob.SetDomain(&Domain);
//    Prob.CheckGradHessian(GrassX);//--
    
    RSGD *RSGDsolver = new RSGD(&Prob, &GrassX);
    RSGDsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
    RSGDsolver->Max_Iteration = 3000;
    RSGDsolver->Initstepsize = 0.002;
//    RSGDsolver->BatchSize = 10;
    RSGDsolver->OutputGap = 20;
    RSGDsolver->Tolerance = 1e-2;
    //RSGDsolver->LMrestart = true;
    RSGDsolver->CheckParams();
    RSGDsolver->Run();
    delete RSGDsolver;
    
    
//    RSVRG *RSVRGsolver = new RSVRG(&Prob, &GrassX);
//    RSVRGsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
//    RSVRGsolver->Max_Iteration = 1000;
//    RSVRGsolver->Initstepsize = 0.01;
////    RSVRGsolver->BatchSize = 10;
//    RSVRGsolver->OutputGap = 10;
//    RSVRGsolver->Tolerance = 1e-6;
//    //RSVRGsolver->LMrestart = true;
//    RSVRGsolver->CheckParams();
//    RSVRGsolver->Run();
//    delete RSVRGsolver;
    
//    return;
    
    RAMSGRAD *RAMSGRADsolver = new RAMSGRAD(&Prob, &GrassX);
    RAMSGRADsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
    RAMSGRADsolver->Max_Iteration = 3000;
//    RAMSGRADsolver->BatchSize = 10;
    RAMSGRADsolver->Initstepsize = 0.02;
    RAMSGRADsolver->OutputGap = 10;
    RAMSGRADsolver->Tolerance = 1e-2;
    //RAMSGRADsolver->LMrestart = true;
    RAMSGRADsolver->CheckParams();
    RAMSGRADsolver->Run();
    delete RAMSGRADsolver;
    
    RADAM *RADAMsolver = new RADAM(&Prob, &GrassX);
    RADAMsolver->Verbose = ITERRESULT;//ITERRESULT;//--- FINALRESULT;
    RADAMsolver->Max_Iteration = 3000;
//    RADAMsolver->BatchSize = 10;
    RADAMsolver->Initstepsize = 0.02;
    RADAMsolver->OutputGap = 10;
    RADAMsolver->Tolerance = 1e-2;
    //RADAMsolver->LMrestart = true;
    RADAMsolver->CheckParams();
    RADAMsolver->Run();
    delete RADAMsolver;
    
    //Vector e1(dim,1);
    //e1.SetToZeros();
    //realdp *e1ptr = e1.ObtainWriteEntireData();
    //e1ptr[0] = 1.;
    //e1.Print();
    //Prob.HessianEta(EucX, e1, &e1).Print("test");
    /* Domain.CheckRetraction(EucX);
     Domain.CheckDiffRetraction(EucX, true);
     Domain.CheckLockingCondition(EucX);
     Domain.CheckcoTangentVector(EucX);
     Domain.CheckIsometryofVectorTransport(EucX);
     Domain.CheckIsometryofInvVectorTransport(EucX);
     Domain.CheckVecTranComposeInverseVecTran(EucX);
     Domain.CheckTranHInvTran(EucX);
     Domain.CheckHaddScaledRank1OPE(EucX);*/
    
//    //Prob.CheckGradHessian(EucX);
//    for (integer i = 1; i <= 2; i++)
//    {
//        realdp lrate = 0.005*i;
//        integer BatchSize = 10;
//        SGD *SGDsolver = new SGD(&Prob, &GrassX);
//        SGDsolver->Verbose = FINALRESULT;//ITERRESULT;//--- FINALRESULT;
//        SGDsolver->Max_Iteration = 10;
//        SGDsolver->OutputGap = 10;
//        SGDsolver->Tolerance = 1e-6;
//        SGDsolver->Initstepsize = lrate;
//        SGDsolver->FreqM = 5*N;
//        SGDsolver->BatchSize = BatchSize;
//        //LRBFGSsolver->LMrestart = true;
//        //LRBFGSsolver->CheckParams();
//        SGDsolver->Run();
//        delete SGDsolver;
//        
//        SVRLRBroyden *SVRLRBroydensolver = new SVRLRBroyden(&Prob, &GrassX);
//        SVRLRBroydensolver->Verbose = FINALRESULT;
//        SVRLRBroydensolver->Max_Iteration = 10;
//        SVRLRBroydensolver->OutputGap = 10;
//        SVRLRBroydensolver->LengthSY = 6;
//        SVRLRBroydensolver->Tolerance = 1e-6;
//        SVRLRBroydensolver->Initstepsize = lrate;
//        SVRLRBroydensolver->FreqM = 5*N;
//        SVRLRBroydensolver->BatchSize = BatchSize;
//        //LRBFGSsolver->LMrestart = true;
//        //LRBFGSsolver->CheckParams();
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
//    }
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
    integer N, n, p, HasHHR;
    n = mxGetM(prhs[1]);
    p = mxGetN(prhs[1]);
    N = mxGetN(prhs[0]);
    
    /*Check the correctness of the inputs*/
    if(mxGetM(prhs[0]) != n )
    {
        mexErrMsgTxt("The size of the data is not correct!\n");
    }
    
    HasHHR = static_cast<integer> (mxGetScalar(prhs[2]));
    
    genrandseed(0);
    
    CheckMemoryDeleted = new std::map<integer *, integer>;
    //    testStieBrockett(B, D, n, p, X, Xopt);
    
    Grassmann Domain(n,p);
    Variable initX = Domain.RandominManifold();
    realdp *initXptr = initX.ObtainWriteEntireData();
    for(integer i = 0; i < n * p; i++)
        initXptr[i] = X[i];
    
    Vector AA(n, N);
    realdp *AAptr = AA.ObtainWriteEntireData();
    for(integer i = 0; i < N * n; i++)
        AAptr[i] = A[i];
    
    GrassPCA Prob(AA,N,n,p);
    Prob.SetDomain(&Domain);
    
    Domain.SetHasHHR(HasHHR != 0);
//    IsIntr = static_cast<integer> (mxGetScalar(prhs[4]));
//    Domain.SetIsIntrApproach(IsIntr != 0);
    //Domain.SetIsIntrApproach(false);
    //Domain.CheckParams();
    
    // Call the function defined in DriverMexProb.h
    ParseSolverParamsAndOptimizing(prhs[3], &Prob, &initX, plhs);
    
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
