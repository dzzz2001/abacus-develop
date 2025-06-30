
#include "test/TestEucQuadratic.h"

using namespace ROPTLIB;

void testEucQuadratic(void)
{
	// size of the domain
	integer dim = 10;
    Vector A(dim, dim);
    A.RandGaussian();
    A = A * A.GetTranspose();
	// Obtain an initial iterate
//	Variable EucX(dim, 1);
//	EucX.RandInManifold();
	Euclidean Domain(dim);
    Variable EucX = Domain.RandominManifold();
	// Define the problem
	EucQuadratic Prob(A);
	Prob.SetDomain(&Domain);
    
//    A.EigenDecomSym();
//    A.Field("_EigVal").Print("eigs:");
    
   /* Domain.CheckRetraction(EucX);
    Domain.CheckVecTranDiffRet(EucX, true);
    Domain.CheckLockingCondition(EucX);
    Domain.CheckVecTranDiffRetAdjoint(EucX);
    Domain.CheckIsometryofVectorTransport(EucX);
    Domain.CheckIsometryofInvVectorTransport(EucX);
    Domain.CheckVecTranComposeInverseVecTran(EucX);
    Domain.CheckTranHInvTran(EucX);
    Domain.CheckHaddScaledRank1OPE(EucX);*/
    
//    Prob.CheckGradHessian(EucX);

    LRTRSR1woR *LRTRSR1woRsolver = new LRTRSR1woR(&Prob, &EucX);
    LRTRSR1woRsolver->Verbose = ITERRESULT;//--- FINALRESULT;
    LRTRSR1woRsolver->Max_Iteration = 1000;
    LRTRSR1woRsolver->initial_Delta = 1e-2;
    LRTRSR1woRsolver->OutputGap = 100;
    LRTRSR1woRsolver->LMrestart = false;
    LRTRSR1woRsolver->LengthSY = 1;
    LRTRSR1woRsolver->CheckParams();
    LRTRSR1woRsolver->Run();
    delete LRTRSR1woRsolver;
    
    LRTRSR1woR *LRTRSR1woRsolver2 = new LRTRSR1woR(&Prob, &EucX);
    LRTRSR1woRsolver2->Verbose = ITERRESULT;//--- FINALRESULT;
    LRTRSR1woRsolver2->Max_Iteration = 1000;
    LRTRSR1woRsolver2->initial_Delta = 1e-2;
    LRTRSR1woRsolver2->OutputGap = 100;
    LRTRSR1woRsolver2->LMrestart = false;
    LRTRSR1woRsolver2->LengthSY = 4;
    LRTRSR1woRsolver2->CheckParams();
    LRTRSR1woRsolver2->Run();
    delete LRTRSR1woRsolver2;

//    LRBFGS *LRBFGSsolver = new LRBFGS(&Prob, &EucX);
//    LRBFGSsolver->Verbose = ITERRESULT;//--- FINALRESULT;
//    LRBFGSsolver->Max_Iteration = 20;
//    LRBFGSsolver->OutputGap = 1;
//    LRBFGSsolver->InitSteptype = LSSM_ONESTEP;
//    LRBFGSsolver->LS_ratio1 = 0.25;
//    LRBFGSsolver->LS_ratio2 = 0.25;
////    LRBFGSsolver->Num_pre_funs = 1000;
//    LRBFGSsolver->LengthSY = 10;
////    LRBFGSsolver->CheckParams();
//    LRBFGSsolver->Run();
//    delete LRBFGSsolver;
//
//    LRTRSR1 *LRTRSR1solver = new LRTRSR1(&Prob, &EucX);
//    LRTRSR1solver->Verbose = ITERRESULT;//--- FINALRESULT;
//    LRTRSR1solver->Max_Iteration = 1000;
////    LRTRSR1solver->LengthSY  = 10;
//    LRTRSR1solver->OutputGap = 1;
//    LRTRSR1solver->CheckParams();
//    LRTRSR1solver->Run();
//    delete LRTRSR1solver;
    
};

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs < 4)
	{
		mexErrMsgTxt("The number of arguments should be at least four.\n");
	}
    
    realdp *B, *X;
	B = mxGetPr(prhs[0]);
	X = mxGetPr(prhs[1]);
	/* dimensions of input matrices */
	integer n, HasHHR;
	n = mxGetM(prhs[0]);
    
	/*Check the correctness of the inputs*/
	if(mxGetN(prhs[0]) != n)
	{
		mexErrMsgTxt("The size of matrix B is not correct.\n");
	}
	if(mxGetM(prhs[1]) != n || mxGetN(prhs[1]) != 1)
	{
		mexErrMsgTxt("The size of the initial X is not correct!\n");
	}
	HasHHR = static_cast<integer> (mxGetScalar(prhs[2]));

	genrandseed(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;
	//	testStieBrockett(B, D, n, p, X, Xopt);

	// Define the manifold
	Euclidean Domain(n);
    Variable initX = Domain.RandominManifold();
    realdp *initXptr = initX.ObtainWriteEntireData();
    for(integer i = 0; i < n; i++)
        initXptr[i] = X[i];

    Vector BB(n, n);
    realdp *BBptr = BB.ObtainWriteEntireData();
    for(integer i = 0; i < n * n; i++)
        BBptr[i] = B[i];
	// Define the Brockett problem
	EucQuadratic Prob(BB);
	Prob.SetDomain(&Domain);

	Domain.SetHasHHR(HasHHR != 0);

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
