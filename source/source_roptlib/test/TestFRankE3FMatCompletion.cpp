#include "test/TestFRankE3FMatCompletion.h"

using namespace ROPTLIB;

void testFRankE3FMatCompletion(void)
{
//    integer m = 50, n = 50, r = 4;
//    integer m = 8, n = 7, r = 2;
    integer m = 8000, n = 8000, r = 30;

//    Vector AA(m, r), BB(r, r);
//    AA.RandGaussian(); BB.RandGaussian();
//    unsigned long startt= getTickCount();
//    AA * BB;
//
//    std::cout << "test:" << static_cast<realdp>(getTickCount() - startt) / CLK_PS << std::endl;//----
//
//    return;
	// Generate the matrices in the matrix completion approximation problem.
	integer dim = (m + n - r) * r;
    integer nz = 3 * dim;
	nz = (nz > m * n) ? m * n : nz;
	unsigned long *ir = new unsigned long[nz * 2];
	unsigned long *jc = ir + nz;
    
	integer *tmpforidx = new integer[m * n];
	for (integer i = 0; i < m * n; i++)
		tmpforidx[i] = i;
	/*nz number of indices*/
	integer idx = 0, itmp;
	for (integer i = 0; i < nz; i++)
	{
		/*idx is an integer in [0, m - i - 1]*/
		idx = static_cast<integer> ((m * n - i) * genrandreal());
		while (idx >= m * n - i)
			idx = static_cast<integer> ((m * n - i) * genrandreal());
		/*the chosen idx is put at the end of the array*/
		itmp = tmpforidx[m * n - i - 1];
		tmpforidx[m * n - i - 1] = tmpforidx[idx];
		tmpforidx[idx] = itmp;
	}
    
    for(integer i = 0; i < nz; i++)
    {
        for(integer j = i + 1; j < nz; j++)
        {
            if(tmpforidx[i] > tmpforidx[j])
            {
                itmp = tmpforidx[i];
                tmpforidx[i] = tmpforidx[j];
                tmpforidx[j] = itmp;
            }
        }
    }
    
//    for(integer i = 0; i < nz; i++)
//    {
//        std::cout << "i:" << tmpforidx[i] << std::endl;//---
//    }
    
	for (integer i = 0; i < nz; i++)
	{
        jc[i] = static_cast<integer> (tmpforidx[i] / m);
        ir[i] = tmpforidx[i] - m * jc[i];
	}
	delete[] tmpforidx;
    
    
    unsigned long *jcc = new unsigned long[n + 1];
    for(integer i = 0; i < n + 1; i++)
        jcc[i] = 0;
    
    for(integer i = 0; i < nz; i++)
    {
        jcc[jc[i] + 1] = i + 1;
    }
    for(integer i = 0; i < n; i++)
    {
        if(jcc[i] > jcc[i + 1])
            jcc[i + 1] = jcc[i];
    }
    
    
//    for(integer i = 0; i < nz; i++)
//        std::cout << "i:" << i << ":" << ir[i] << ":" << jc[i] << std::endl;//---
//
//    for(integer i = 0; i < n + 1; i++)
//        std::cout << "i:" << i << ":" << jcc[i] << std::endl;//---
    
	integer mr = m * r, nr = n * r;
	realdp *A_U = new realdp[mr];
	realdp *A_V = new realdp[nr];
	for (integer i = 0; i < m * r; i++)
	{
		A_U[i] = genrandnormal();
	}
	for (integer i = 0; i < n * r; i++)
	{
		A_V[i] = genrandnormal();
	}
	realdp *V = new realdp[nz];
	for (integer i = 0; i < nz; i++)
	{
		V[i] = 0;
		for (integer j = 0; j < r; j++)
		{
			V[i] += A_U[ir[i] + j * m] * A_V[jc[i] + j * n];
		}
	}
	delete[]A_U;
	delete[]A_V;

    FixedRankE3F Domain(m, n, r);
//    Domain.SetHasHHR(true);
    Variable InitialX = Domain.RandominManifold();
    
//    Domain.GetEMPTYINTR().Print("EMPTYINTR:");//---
            
//    Domain.CheckIntrExtr(InitialX);
//    Domain.CheckRetraction(InitialX);
//    Domain.CheckVecTranDiffRet(InitialX, false);
//    Domain.CheckVecTranDiffRetAdjoint(InitialX);
//    Domain.CheckIsometryofVectorTransport(&InitialX);
//
//    Domain.CheckLockingCondition(&InitialX);
//    Domain.CheckIsometryofInvVectorTransport(&InitialX);
//    Domain.CheckVecTranComposeInverseVecTran(&InitialX);
//    Domain.CheckTranHInvTran(&InitialX);
//    return;
    
//    InitialX.Print("initX:");//---
    
    
    FRankE3FMatCompletion Prob(ir, jc, jcc, V, nz, m, n, r);
	Prob.SetDomain(&Domain);
//    Domain.SetHasHHR(true);
    Domain.CheckParams();
//    Prob.SetNumGradHess(true);
    
	// test LRBFGS
	//printf("********************************Check LRBFGS*************************************\n");
    LRBFGS LRBFGSsolver(&Prob, &InitialX);
//    LRBFGS LRBFGSsolver(&Prob, &InitialX);
	LRBFGSsolver.OutputGap = 1;
	LRBFGSsolver.Max_Iteration = 100;
    LRBFGSsolver.Verbose = ITERRESULT; //-- ITERRESULT;//--- FINALRESULT;
									    //LRBFGSsolver.CheckParams();
	LRBFGSsolver.Tolerance = static_cast<realdp> (1e-6);
    LRBFGSsolver.LengthSY = 4;
//    LRBFGSsolver.LMrestart = false;
//    LRBFGSsolver.InitSteptype = LSSM_ONESTEP;
    LRBFGSsolver.CheckParams();
	LRBFGSsolver.Run();

//    // test RCG
//    //printf("********************************Check LRBFGS*************************************\n");
//    RCG RCGsolver(&Prob, &InitialX);
//    RCGsolver.RCGmethod = POLAK_RIBIERE_MOD; //FLETCHER_REEVES, POLAK_RIBIERE_MOD, HESTENES_STIEFEL, FR_PR, DAI_YUAN, HAGER_ZHANG, RCGMETHODSLENGTH };
//    RCGsolver.OutputGap = 1;
//    RCGsolver.LineSearch_LS = LSSM_INPUTFUN;
//    RCGsolver.LinesearchInput = &LRMatrixCompletionLinesearchInput;
//    RCGsolver.Max_Iteration = 200;
//    RCGsolver.Verbose = ITERRESULT; //-- ITERRESULT;//--- FINALRESULT;
//                                        //LRBFGSsolver.CheckParams();
//    RCGsolver.Tolerance = static_cast<realdp> (1e-6);
//    RCGsolver.CheckParams();
//    RCGsolver.Run();
    
//    Prob.CheckGradHessian(LRBFGSsolver.GetXopt());

//        //printf("********************************Check LRBFGS*************************************\n");
//        LRBFGS LRBFGSsolver2(&Prob, &InitialX);
//    //    LRBFGS LRBFGSsolver(&Prob, &InitialX);
//        LRBFGSsolver2.OutputGap = 100;
//        LRBFGSsolver2.Max_Iteration = 500;
////        LRBFGSsolver2.LineSearch_LS = LSSM_WOLFE;
//        LRBFGSsolver2.Verbose = ITERRESULT; //-- ITERRESULT;//--- FINALRESULT;
//                                            //LRBFGSsolver.CheckParams();
//        LRBFGSsolver2.Tolerance = static_cast<realdp> (1e-6);
//        LRBFGSsolver2.LengthSY = 4;
//        LRBFGSsolver2.LMrestart = true;
//    //    LRBFGSsolver2.CheckParams();
//        LRBFGSsolver2.Run();

//    Prob.MinMaxEigValHess(LRBFGSsolver.GetXopt()).Print("eigs:");//---
//    Prob.CheckGradHessian(LRBFGSsolver.GetXopt());

//	if (LRBFGSsolver.Getnormgfgf0() < 1e-6)
//		printf("SUCCESS!\n");
//	else
//		printf("FAIL!\n");

//    LRTRSR1 LRTRSR1Ssolver(&Prob, &InitialX);
//    LRTRSR1Ssolver.initial_Delta = 100;
//    LRTRSR1Ssolver.OutputGap = 100;
//    LRTRSR1Ssolver.Max_Iteration = 500;
//    LRTRSR1Ssolver.LMrestart = false;
//    LRTRSR1Ssolver.LengthSY = 4;
//    LRTRSR1Ssolver.Verbose = ITERRESULT; //--- ITERRESULT;//--- FINALRESULT;
//                                     //LRBFGSsolver.CheckParams();
//    LRTRSR1Ssolver.Tolerance = static_cast<realdp> (1e-6);
////    LRTRSR1Ssolver.CheckParams();
//    LRTRSR1Ssolver.Run();
//
//    LRTRSR1 LRTRSR1Ssolver2(&Prob, &InitialX);
//    LRTRSR1Ssolver2.initial_Delta = 100;
//    LRTRSR1Ssolver2.OutputGap = 100;
//    LRTRSR1Ssolver2.Max_Iteration = 500;
//    LRTRSR1Ssolver2.LMrestart = true;
//    LRTRSR1Ssolver2.LengthSY = 4;
//    LRTRSR1Ssolver2.Verbose = ITERRESULT; //--- ITERRESULT;//--- FINALRESULT;
//                                     //LRBFGSsolver.CheckParams();
//    LRTRSR1Ssolver2.Tolerance = static_cast<realdp> (1e-6);
////    LRTRSR1Ssolver2.CheckParams();
//    LRTRSR1Ssolver2.Run();

    
//    Variable LRTRSR1Xopt = LRTRSR1Ssolver2.GetXopt();
    
//    Vector H = LRTRSR1Xopt.GetElement(0), G = LRTRSR1Xopt.GetElement(1);
//    H.QRDecom(); G.QRDecom();
//    Vector HR = H.Field("_R"), HG = H.Field("_R");
//    Vector tmp = HR * HG.GetTranspose();
//    tmp.SVDDecom();
//    tmp.Field("_S").Print("S:");
////    HR.Print("HR:");
////    HG.Print("HG:");

//    printf("********************************Check LRBFGS*************************************\n");
//    LRBFGS LRBFGSsolver2(&Prob, &LRTRSR1Xopt);
//    LRBFGSsolver2.OutputGap = 1;
//    LRBFGSsolver2.Max_Iteration = 500;
//    LRBFGSsolver2.Verbose = FINALRESULT; //-- ITERRESULT;//--- FINALRESULT;
//                                     //LRBFGSsolver.CheckParams();
//    LRBFGSsolver2.Tolerance = static_cast<realdp> (1e-6);
//    LRBFGSsolver2.LengthSY = 0;
////    LRBFGSsolver2.LMrestart = true;
//    LRBFGSsolver2.Run();
    
	delete[] V;
	delete[] ir;
    delete [] jcc;
};

/*We don't have to a line search algorithm defined in the solvers. The line seach algorithm can be defined here.
 This is the initial step size proposed in Bart's paper.*/
//realdp LRMatrixCompletionLinesearchInput(integer iter, Variable *x1, Vector *eta1, realdp initialstepsize, realdp initialslope, const Problem *prob, const Solvers *solver)
realdp LRMatrixCompletionLinesearchInput(integer iter, const Variable &x1, const Vector &eta1, realdp initialstepsize, realdp initialslope, const Problem *prob, const Solvers *solver)
{
    const FRankE3FMatCompletion *P = dynamic_cast<FRankE3FMatCompletion *> (const_cast<Problem*> (prob));

    const realdp *UDVtmAptr = (realdp *) x1.GetSparseMatrixinFields()->GetVals();
    
//    Vector UDVtmA = x1.Field("UDVtmA");
//    const realdp *UDVtmAptr = UDVtmA.ObtainReadData();
    unsigned long *ir = P->ir, *jc = P->jc, *jcc = P->jcc;
    integer m = P->m, n = P->n, r = P->r, nz = P->nz;
    
    const realdp *Uptr = x1.GetElement(0).ObtainReadData();
    const realdp *Dptr = x1.GetElement(1).ObtainReadData();
    const realdp *Vptr = x1.GetElement(2).ObtainReadData();
    
    const realdp *DUptr = eta1.GetElement(0).ObtainReadData();
    const realdp *DDptr = eta1.GetElement(1).ObtainReadData();
    const realdp *DVptr = eta1.GetElement(2).ObtainReadData();
    
    realdp *tmp = new realdp[2 * nz];
    realdp *POmegaeta = tmp + nz;
    FRankE3FMatCompletion::ProjecOmegaUDVT(DUptr, Dptr, Vptr, m, n, r, ir, jc, jcc, nz, POmegaeta);
    FRankE3FMatCompletion::ProjecOmegaUDVT(Uptr, DDptr, Vptr, m, n, r, ir, jc, jcc, nz, tmp);
    axpy_(const_cast<integer *> (&nz), &GLOBAL::DONE, tmp, &GLOBAL::IONE, POmegaeta, &GLOBAL::IONE);
    FRankE3FMatCompletion::ProjecOmegaUDVT(Uptr, Dptr, DVptr, m, n, r, ir, jc, jcc, nz, tmp);
    axpy_(const_cast<integer *> (&nz), &GLOBAL::DONE, tmp, &GLOBAL::IONE, POmegaeta, &GLOBAL::IONE);
    
    realdp denor, nume;

    nume = dot_(&nz, const_cast<realdp *> (UDVtmAptr), &GLOBAL::IONE, POmegaeta, &GLOBAL::IONE);
    denor = dot_(&nz, POmegaeta, &GLOBAL::IONE, POmegaeta, &GLOBAL::IONE);

    delete[] tmp;
    return (-nume / denor < 0) ? 1 : -nume / denor;
}

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//    integer m = 8000, n = 8000, r = 80;
//
//    Vector AA(m, r), BB(r, r), CC(m, r);
//    AA.RandGaussian(); BB.RandGaussian();
//    realdp *AAptr = AA.ObtainWritePartialData();
//    realdp *BBptr = BB.ObtainWritePartialData();
//    realdp *CCptr = CC.ObtainWritePartialData();
//    unsigned long startt= getTickCount();
//    gemm_(GLOBAL::N, GLOBAL::N, &m, &r, &r, &GLOBAL::DONE, AAptr, &m, BBptr, &r, &GLOBAL::DZERO, CCptr, &m);
//    std::cout << "test:" << static_cast<realdp>(getTickCount() - startt) / CLK_PS << std::endl;//----
//
//    mexErrMsgTxt("The number of arguments should be at least five.\n");
//    return;
    
	if (nrhs < 5)
	{
		mexErrMsgTxt("The number of arguments should be at least five.\n");
	}
	realdp *A, *X;
	A = mxGetPr(prhs[0]);
	X = mxGetPr(prhs[1]);
	/* dimensions of input matrices */
	integer m, n, HasHHR, nzmax, r;
	size_t *ir, *jc;
	nzmax = mxGetNzmax(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
	r = static_cast<integer> (mxGetScalar(prhs[2]));

	/*Check the correctness of the inputs*/
	if (mxGetM(prhs[1]) != (m + n + r) * r || mxGetN(prhs[1]) != 1)
	{
		mexErrMsgTxt("The size of the initial X is not correct!\n");
	}
	HasHHR = static_cast<integer> (mxGetScalar(prhs[3]));

	genrandseed(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;

	// Define the manifold
	FixedRankE3F Domain(m, n, r);

    Variable initX = Domain.RandominManifold();
    realdp *initXptr = initX.ObtainWriteEntireData();
    for(integer i = 0; i < initX.Getlength(); i++)
    {
        initXptr[i] = X[i];
    }
    
//    for(integer i = 0; i < n + 1; i++)
//    {
//        std::cout << jc[i] << std::endl;//---
//    }
    
	// Define the matrix completion problem
	unsigned long *inir = new unsigned long[2 * nzmax + n + 1];
    unsigned long *injc = inir + nzmax;
    unsigned long *injcc = injc + nzmax;

	for (integer i = 0; i < n; i++)
	{
		for (unsigned long long j = jc[i]; j < jc[i + 1]; j++)
		{
			/*row: ir[j], column: i, entry: A[j]*/
			inir[j] = ir[j];
			injc[j] = i;
		}
        injcc[i] = jc[i];
	}
    injcc[n] = jc[n];

    FRankE3FMatCompletion Prob(inir, injc, injcc, A, nzmax, m, n, r);
    Prob.SetDomain(&Domain);
    Domain.SetIsIntrApproach(true);

	Domain.SetHasHHR(HasHHR != 0);
    
	// Call the function defined in DriverMexProb.h
	ParseSolverParamsAndOptimizing(prhs[4], &Prob, &initX, plhs, &LRMatrixCompletionLinesearchInput);
    
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
