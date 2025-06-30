
#include "test/TestStieSoftICA.h"

using namespace ROPTLIB;

void testStieSoftICA(void)
{
    // size of the Stiefel manifold
    integer n = 12, p = 6;
    // number of covariance matrices
    integer N = 100;

    // Generate the matrices in the joint diagonalization (JD) problem (Soft ICA problem).
    // Use the same approach as the experiments in paper "A Riemannian symmetric rank-one trust-region method".
    Vector C(n, n);
    Vector Cs(1, &C, N);
    realdp *Csptr = Cs.ObtainWriteEntireData();
    
    for (integer i = 0; i < N; i++)
    {
        for (integer j = 0; j < n; j++)
        {
            for (integer k = 0; k < n; k++)
            {
                Csptr[i * n * n + j * n + k] = genrandnormal();
            }
        }
        for (integer j = 0; j < n; j++)
        {
            for (integer k = j; k < n; k++)
            {
                Csptr[i * n * n + j * n + k] += Csptr[i * n * n + k * n + j];
                Csptr[i * n * n + k * n + j] = Csptr[i * n * n + j * n + k];
            }
        }
        for (integer j = 0; j < n * n; j++)
        {
            Csptr[i * n * n + j] *= static_cast<realdp> (0.1);
        }
        for (integer j = 0; j < n; j++)
        {
            Csptr[i * n * n + j * n + j] += n - j;
        }
    }
    
    Stiefel Domain(n, p);
    Domain.ChooseParamsSet1();
    Vector InitialX = Domain.RandominManifold();
    StieSoftICA Prob(Cs, p);
    Prob.SetDomain(&Domain);
    

	Domain.CheckParams();

	//Domain.CheckIntrExtr(&StieX);
	//Domain.CheckRetraction(&StieX);
	//Domain.CheckVecTranDiffRet(&StieX, true);
	//Domain.CheckLockingCondition(&StieX);
	//Domain.CheckVecTranDiffRetAdjoint(&StieX);
	//Domain.CheckIsometryofVectorTransport(&StieX);
	//Domain.CheckIsometryofInvVectorTransport(&StieX);
	//Domain.CheckVecTranComposeInverseVecTran(&StieX);
	//Domain.CheckTranHInvTran(&StieX);
	//Domain.CheckHaddScaledRank1OPE(&StieX);

//	//// Check gradient and Hessian
//	Prob.CheckGradHessian(InitialX);

	// test LRBFGS
	//printf("********************************Check all line search algorithm in LRBFGS*************************************\n");
    LRBFGS *LRBFGSsolver = new LRBFGS(&Prob, &InitialX);
    LRBFGSsolver->LineSearch_LS = LSSM_ARMIJO;
    LRBFGSsolver->Verbose = ITERRESULT; //ITERRESULT;//
    //LRBFGSsolver->InitSteptype = ONESTEP;
    LRBFGSsolver->Max_Iteration = 1000;
    LRBFGSsolver->OutputGap = 100;
    //LRBFGSsolver->CheckParams();
    LRBFGSsolver->Run();
    if (LRBFGSsolver->Getnormgfgf0() < 1e-6)
        printf("SUCCESS!\n");
    else
        printf("FAIL!\n");
    
//    Prob.CheckGradHessian(LRBFGSsolver->GetXopt());
    delete LRBFGSsolver;
}

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 7)
    {
        mexErrMsgTxt("The number of arguments should be at least seven.\n");
    }
	realdp *Cs, *X, *Xopt;
	integer p, n, N, HasHHR, ParamSet;
	Cs = mxGetPr(prhs[0]);
	X = mxGetPr(prhs[1]);
	n = static_cast<integer> (mxGetScalar(prhs[2]));
	p = static_cast<integer> (mxGetScalar(prhs[3]));
	HasHHR = static_cast<integer> (mxGetScalar(prhs[4]));
	ParamSet = static_cast<integer> (mxGetScalar(prhs[5]));
	/* dimensions of input matrices */
	const mwSize *ptrdims = mxGetDimensions(prhs[0]);
	N = ptrdims[2];
    
    if(ptrdims[1] != n || ptrdims[0] != n)
    {
        mexErrMsgTxt("The size of matrix C is not correct.\n");
    }

	if (mxGetM(prhs[1]) != n || mxGetN(prhs[1]) != p)
	{
		mexErrMsgTxt("The size of the initial X is not correct!\n");
	}
    
	printf("(n, p, N):%d, %d, %d\n", n, p, N);

	///*create output matrix*/
	//plhs[0] = mxCreateDoubleMatrix(n, p, mxREAL);
	//Xopt = mxGetPr(plhs[0]);

	genrandseed(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;
	//testStieSoftICA(Cs, n, p, N, X, Xopt);
    
    Vector C(n, n);
    Vector Css(1, &C, N);
    realdp *Csptr = Css.ObtainWriteEntireData();
    for(integer i = 0; i < n * n * N; i++)
        Csptr[i] = Cs[i];
    
    Stiefel Domain(n, p);
    Vector InitialX = Domain.RandominManifold();
    realdp *InitialXptr = InitialX.ObtainWriteEntireData();
    for(integer i = 0; i < n * p; i++)
        InitialXptr[i] = X[i];
    
    StieSoftICA Prob(Css, p);
    Prob.SetDomain(&Domain);
    
	if (ParamSet == 1)
		Domain.ChooseParamsSet1();
	else
    if (ParamSet == 2)
		Domain.ChooseParamsSet2();
	else
    if (ParamSet == 3)
		Domain.ChooseParamsSet3();
	else
    if (ParamSet == 4)
        Domain.ChooseParamsSet4();
    else
    if (ParamSet == 5)
        Domain.ChooseParamsSet5();
	else
    {
        printf("error: ParamSet parameter!");
        return;
    }
	Domain.SetHasHHR((HasHHR != 0));
	//Domain.CheckParams();
    
	ParseSolverParamsAndOptimizing(prhs[6], &Prob, &InitialX, plhs);
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
