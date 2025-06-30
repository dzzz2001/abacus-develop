#include "test/TestCFRankQ2FBlindDecon2D.h"

#ifdef ROPTLIB_WITH_FFTW

using namespace ROPTLIB;

void testCFRankQ2FBlindDecon2D(void)
{
//    std::cout << "h1" << std::endl;//---
	integer n1 = 4, n2 = 8, r = 1, L = n1 * n2;
    
//    std::cout << "h2" << std::endl;//---
    CFixedRankQ2F Domain(L, L, r);
	Domain.SetHasHHR(false);
//    std::cout << "h3" << std::endl;//---
//    std::cout << "L:" << L << std::endl;//----
	Variable InitialX = Domain.RandominManifold();
//    std::cout << "h4" << std::endl;//---
	//Domain.CheckParams();

//    Domain.ChooseParamsSet2();
//	Domain.CheckIntrExtr(InitialX);
//	Domain.CheckRetraction(&InitialX);
//	Domain.CheckVecTranDiffRetAdjoint(&InitialX);
//	Domain.CheckVecTranDiffRet(&InitialX, false);
//	Domain.CheckIsometryofVectorTransport(&InitialX);
//
//	Domain.CheckLockingCondition(&InitialX);
//	Domain.CheckIsometryofInvVectorTransport(&InitialX);
//	Domain.CheckVecTranComposeInverseVecTran(&InitialX);
//	Domain.CheckTranHInvTran(&InitialX);

//	return;

	//InitialX.Print("initialX:");

    Vector y(L, "complex");
//    std::cout << "ylength:" << y.Getlength() << std::endl;//-----
//    std::cout << "h5" << std::endl;//---
    y.RandGaussian();
//    std::cout << "h6" << std::endl;//---
	// Generate the matrices in the Low rank approximation problem.
    realdp *B = new realdp[L * L * 2 + L * L * 2];
	realdp *C = B + L * L * 2;
	for (integer i = 0; i < L * L * 2 + L * L * 2; i++)
		B[i] = genrandnormal();
	integer nzmaxB = L * L;
	integer nzmaxC = L * L;
	unsigned long *irB = new unsigned long[2 * L * L + 2 * L * L + 2 * (L + 1)];
	unsigned long *jcB = irB + L * L;
	unsigned long *irC = jcB + L * L;
	unsigned long *jcC = irC + L * L;
    unsigned long *jccB = jcC + L * L;
    unsigned long *jccC = jccB + L + 1;
	for (integer i = 0; i < L; i++)
	{
		for (integer j = 0; j < L; j++)
		{
			irB[j + i * L] = j;
			jcB[j + i * L] = i;
		}
	}
	for (integer i = 0; i < L; i++)
	{
		for (integer j = 0; j < L; j++)
		{
			irC[j + i * L] = j;
			jcC[j + i * L] = i;
		}
	}
    for(integer i = 0; i < L + 1; i++)
    {
        jccB[i] = i * L;
        jccC[i] = i * L;
    }
    
    SparseMatrix sB(L, L, irB, jcB, jccB, (realdpcomplex *) B, nzmaxB);
    SparseMatrix sC(L, L, irC, jcC, jccC, (realdpcomplex *) C, nzmaxC);

//    y.Print("y:");//---
//    sB.Print("sB:");//---
//    sC.Print("sC:");//---
    
    realdp rho = 0;
    CFRankQ2FBlindDecon2D Prob(y, sB, sC, n1, n2, r, rho, 1, 1);
    
	Prob.SetDomain(&Domain);
    Domain.CheckParams();
    
//	Prob.CheckGradHessian(InitialX);

	LRBFGS *RSDsolver = new LRBFGS(&Prob, &InitialX);
//	RSD *RSDsolver = new RSD(&Prob, &InitialX);
	//->LineSearch_LS = ARMIJO;
	//RSDsolver->LS_beta = 0.01;
	//RSDsolver->RCGmethod = DAI_YUAN;
    RSDsolver->Verbose = ITERRESULT; //-- FINALRESULT;
	RSDsolver->OutputGap = 1;
	RSDsolver->Max_Iteration = 200;
	RSDsolver->Accuracy = 1e-6;
	RSDsolver->Finalstepsize = 1;
	RSDsolver->Tolerance = 1e-6;
//	RSDsolver->LengthSY = 0;
//	RSDsolver->nu = 0;
	RSDsolver->LS_ratio1 = 0.3;
	RSDsolver->LS_ratio2 = 0.3;
//	RSDsolver->InitSteptype = LSSM_ONESTEP;
	//RSDsolver->CheckParams();
	RSDsolver->Run();
	//Prob.CheckGradHessian(&InitialX);//--
//	Prob.CheckGradHessian(RSDsolver->GetXopt());//--
	if (RSDsolver->Getnormgfgf0() < 1e-6)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

//    Prob.CheckGradHessian(RSDsolver->GetXopt());
//    Prob.MinMaxEigValHess(RSDsolver->GetXopt()).Print("eigs:");//--
	delete RSDsolver;
    
    sB.Setnullptr();
    sC.Setnullptr();
	delete[] B;
	delete[] irB;
};

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ /*TestCFRankQ2FBlindDecon2D(y, B, C, Xinitial, n1, n2, r, HasHHR, SolverParams, rho, d, mu);*/
	if (nrhs < 12)
	{
		mexErrMsgTxt("The number of arguments should be at least twelve.\n");
	}
	realdp *y, *B, *C, *X;
    y = (realdp *) mxGetComplexDoubles(prhs[0]);
    B = (realdp *) mxGetComplexDoubles(prhs[1]);
    C = (realdp *) mxGetComplexDoubles(prhs[2]);
//    std::cout << "h1:" << std::endl;//---
	X = (realdp *) mxGetComplexDoubles(prhs[3]);
	/* dimensions of input matrices */
	integer L, K, N, HasHHR, n1, n2, r, ParamSet;
	L = mxGetM(prhs[0]);
	K = mxGetN(prhs[1]);
	N = mxGetN(prhs[2]);
	if (K != L || mxGetM(prhs[1]) != L || mxGetM(prhs[2]) != L || N != L)
	{
		mexErrMsgTxt("The size of B or the size of C is not correct.\n");
	}
//    std::cout << "h2:" << std::endl;//---
	n1 = static_cast<integer> (mxGetScalar(prhs[4]));
	n2 = static_cast<integer> (mxGetScalar(prhs[5]));
	r = static_cast<integer> (mxGetScalar(prhs[6]));
    HasHHR = static_cast<integer> (mxGetScalar(prhs[7]));
	realdp rho, d, mu;
	rho = mxGetScalar(prhs[8]);
	d = mxGetScalar(prhs[9]);
	mu = mxGetScalar(prhs[10]);
    ParamSet = static_cast<integer> (mxGetScalar(prhs[11]));
	if (mxGetM(prhs[3]) != 2 * L * r)
	{
		mexErrMsgTxt("The size of initial x is not correct.\n");
	}
    
//    std::cout << "h3:" << std::endl;//---
	bool isBsparse = mxIsSparse(prhs[1]);
	bool isCsparse = mxIsSparse(prhs[2]);
	integer nzmaxB = 0, nzmaxC = 0;
	size_t *irB = nullptr, *jcB = nullptr, *irC = nullptr, *jcC = nullptr;
	unsigned long *inirB = nullptr, *injcB = nullptr, *inirC = nullptr, *injcC = nullptr, *injccB = nullptr, *injccC = nullptr;
    
    
	if (isBsparse)
	{
		nzmaxB = mxGetNzmax(prhs[1]);
		irB = mxGetIr(prhs[1]);
		jcB = mxGetJc(prhs[1]);

		inirB = new unsigned long[2 * nzmaxB + K + 1];
		injcB = inirB + nzmaxB;
        injccB = injcB + nzmaxB;
		for (integer i = 0; i < K; i++)
		{
			for (unsigned long long j = jcB[i]; j < jcB[i + 1]; j++)
			{
				/*row: ir[j], column: i, entry: A[j]*/
				inirB[j] = irB[j];
				injcB[j] = i;
			}
            injccB[i] = jcB[i];
		}
        injccB[K] = jcB[K];
//        for(integer i = 0; i < 2 * nzmaxB; i++)
//        {
//            if(abs(B[i]) < 1e-10)
//                B[i] = 0;
//        }
	}
    
//    std::cout << "h4:" << std::endl;//---
	if (isCsparse)
	{
		nzmaxC = mxGetNzmax(prhs[2]);
		irC = mxGetIr(prhs[2]);
		jcC = mxGetJc(prhs[2]);

		inirC = new unsigned long[2 * nzmaxC + N + 1];
		injcC = inirC + nzmaxC;
        injccC = injcC + nzmaxC;
		for (integer i = 0; i < N; i++)
		{
			for (unsigned long long j = jcC[i]; j < jcC[i + 1]; j++)
			{
				/*row: ir[j], column: i, entry: A[j]*/
				inirC[j] = irC[j];
				injcC[j] = i;
			}
            injccC[i] = jcC[i];
		}
        injccC[N] = jcC[N];
//        for(integer i = 0; i < 2 * nzmaxC; i++)
//        {
//            if(abs(B[i]) < 1e-10)
//                C[i] = 0;
//        }
	}
//    std::cout << "h5:" << std::endl;//---

	genrandseed(0);
	CheckMemoryDeleted = new std::map<integer *, integer>;

	// Define the manifold
	CFixedRankQ2F Domain(K, N, r);
    
    if (ParamSet == 1)
        Domain.ChooseParamsSet1();
    else
        Domain.ChooseParamsSet2();
    
    SparseMatrix sB(L, K, inirB, injcB, injccB, (realdpcomplex *) B, nzmaxB);
    SparseMatrix sC(L, N, inirC, injcC, injccC, (realdpcomplex *) C, nzmaxC);
//    sB.Print("sB:");//--
//    sC.Print("sC:");//--
//    std::cout << "h6:" << std::endl;//---
    Vector yy(L, "complex");
    realdp *yyptr = yy.ObtainWriteEntireData();
    for(integer i = 0; i < 2 * L; i++)
        yyptr[i] = y[i];
    
//    std::cout << "h7:" << std::endl;//---
    CFRankQ2FBlindDecon2D Prob(yy, sB, sC, n1, n2, r, rho, d, mu);
//    std::cout << "h71:" << std::endl;//---
    Prob.SetDomain(&Domain);
//    std::cout << "h72:" << std::endl;//---
    Variable LRX = Domain.RandominManifold();
//    std::cout << "h73:" << std::endl;//---
    realdp *LRXptr = LRX.ObtainWriteEntireData();
//    std::cout << "h74:" << std::endl;//---
    for(integer i = 0; i < LRX.Getlength(); i++)
        LRXptr[i] = X[i];
    
//    std::cout << 2 * L * r << ":" << LRX.Getlength() << std::endl;//----
    
//    sB.Print("sB:");//---
//    sC.Print("sC:");//---
//    yy.Print("y:");//--
//    LRX.Print("initX:");//---
    
//    std::cout << "f:" << Prob.f(LRX) << std::endl;
    
//    std::cout << "h8:" << std::endl;//---
	Domain.SetHasHHR(HasHHR != 0);
//    yy.Print("y:");//--
//    sB.Print("sB:");//---
//    sC.Print("sC:");//---
//    LRX.Print("LRX:");//---
	// Call the function defined in DriverMexProb.h
	ParseSolverParamsAndOptimizing(prhs[12], &Prob, &LRX, plhs);
//    std::cout << "h9:" << std::endl;//---

	std::map<integer *, integer>::iterator iter = CheckMemoryDeleted->begin();
	for (iter = CheckMemoryDeleted->begin(); iter != CheckMemoryDeleted->end(); iter++)
	{
		if (iter->second != 1)
			printf("Global address: %p, sharedtimes: %d\n", iter->first, iter->second);
	}
	delete CheckMemoryDeleted;
    
    sB.Setnullptr();
    sC.Setnullptr();
	delete[] inirB;
	delete[] inirC;
//	delete[] y;
//	delete[] B;
//	delete[] C;

	return;
}

#endif
#endif
