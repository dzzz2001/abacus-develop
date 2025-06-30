
#include "test/TestGrassSVPCA.h"

using namespace ROPTLIB;

void testGrassSVPCA(void)
{
	// size of the Stiefel manifold
//    integer n = 5, m = 4, p = 3;
    integer n = 256, m = 100, p = 4;
    realdp lambda = 1;

	// Generate the matrices in the SPCA problem.
    Vector A(m, n);
    realdp *Aptr = A.ObtainWriteEntireData();
	/*B is an n by n matrix*/
	for (integer i = 0; i < n * m; i++)
	{
        Aptr[i] = genrandnormal();
	}
    for(integer i = 0; i < n; i++)
    {
        realdp s = 0;
        for(integer j = 0; j < m; j++)
        {
            s += Aptr[j + i * m];
        }
        s /= m;
        for(integer j = 0; j < m; j++)
            Aptr[j + i * m] -= s;
        s = 0;
        for(integer j = 0; j < m; j++)
        {
            s += Aptr[j + i * m] * Aptr[j + i * m];
        }
        s = std::sqrt(s);
        for(integer j = 0; j < m; j++)
            Aptr[j + i * m] /= s;
    }
    
    
    // Define the manifold
//    integer numoftypes = 1, numofmani = p;
//    Sphere mani(n);
//    mani.ChooseParamsSet3();
//    ProductManifold Domain(numoftypes, &mani, numofmani);
//    Domain.SetIsIntrApproach(false);
    Grassmann Domain(n, p);
    Domain.ChooseParamsSet2();
    
    Variable GrassX = Domain.RandominManifold();

//    Domain.CheckVecTranDiffRet(GrassX, false);
//    Domain.CheckVecTranDiffRetAdjoint(GrassX);
//    Domain.CheckInverseVecTranDiffRet(GrassX);
//    Domain.CheckInverseVecTranDiffRetAdjoint(GrassX);
//    return;
    
    // Define the SPCA problem
    GrassSVPCA Prob(A, lambda, n, m, p); // DIAGRHESS // LIPSCHITZ
    /*The domain of the problem is a Stiefel manifold*/
    Prob.SetDomain(&Domain);
    Domain.CheckParams();
    
//    IARPG *IARPGsolver = new IARPG(&Prob, &GrassX);
//    IARPGsolver->Max_Iteration = 100;
//    IARPGsolver->Verbose = ITERRESULT;
//    IARPGsolver->Variant = LSPG_REGULAR; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
//    IARPGsolver->CheckParams();
//    IARPGsolver->Run();
//
//    IARPGsolver->GetXopt().Print("Xopt:");
//    delete IARPGsolver;
    
    IRPG *IRPGsolver = new IRPG(&Prob, &GrassX);
    IRPGsolver->Max_Iteration = 200;
    IRPGsolver->Verbose = ITERRESULT;
    IRPGsolver->OutputGap = 30;
    IRPGsolver->ProxMapType = LSPG_GLOBAL; //-- LSPG_UNILIMIT; LSPG_GLOBAL
    IRPGsolver->Variant = LSPG_BB; //-- LSPG_BB; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    IRPGsolver->CheckParams();
    IRPGsolver->Run();
    delete IRPGsolver;
    
    IRPG *IRPGsolver2 = new IRPG(&Prob, &GrassX);
    IRPGsolver2->Max_Iteration = 200;
    IRPGsolver2->Verbose = ITERRESULT;
    IRPGsolver2->OutputGap = 30;
    IRPGsolver2->ProxMapType = LSPG_UNILIMIT; //-- LSPG_UNILIMIT; LSPG_GLOBAL
    IRPGsolver2->Variant = LSPG_BB; //-- LSPG_BB; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    IRPGsolver2->CheckParams();
    IRPGsolver2->Run();
    delete IRPGsolver2;
    
    IRPG *IRPGsolver3 = new IRPG(&Prob, &GrassX);
    IRPGsolver3->Max_Iteration = 200;
    IRPGsolver3->Verbose = ITERRESULT;
    IRPGsolver3->OutputGap = 30;
    IRPGsolver3->ProxMapType = LSPG_LOCAL; //-- LSPG_UNILIMIT; LSPG_GLOBAL; LSPG_LOCAL
    IRPGsolver3->Variant = LSPG_BB; //-- LSPG_BB; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    IRPGsolver3->CheckParams();
    IRPGsolver3->Run();
    delete IRPGsolver3;
}

/*If it is compiled in Matlab, then the following "mexFunction" is used as the entrance.*/
#ifdef MATLAB_MEX_FILE

/*Help to check the memory leakage problem. No necesary any more.*/
std::map<integer *, integer> *CheckMemoryDeleted;

/*This function checks the number and formats of input parameters.
nlhs: the number of output in mxArray format
plhs: the output objects in mxArray format
nrhs: the number of input in mxArray format
prhs: the input objects in mxArray format */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ /* (A, lambda, Xinitial, SolverParams) */
	if(nrhs < 4)
	{
		mexErrMsgTxt("The number of arguments should be at least four.\n");
	}
	realdp *A, *X;
    realdp lambda;
	A = mxGetPr(prhs[0]);
    lambda = static_cast<realdp> (mxGetScalar(prhs[1]));
	X = mxGetPr(prhs[2]);
	/* dimensions of input matrices */
	integer m, n, p;
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
    p = mxGetN(prhs[2]);

	/*Check the correctness of the inputs*/
	if(mxGetM(prhs[2]) != n)
	{
		mexErrMsgTxt("The size of matrix Xinit is not correct.\n");
	}

	genrandseed(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;

    Vector AA(m, n);
    realdp *AAptr = AA.ObtainWriteEntireData();
    for(integer i = 0; i < m * n; i++)
        AAptr[i] = A[i];
    
    // Define the manifold
    Grassmann Domain(n, p);
    Domain.ChooseParamsSet2();

	// Obtain an initial iterate by taking the Q factor of qr decomposition
	Variable initX = Domain.RandominManifold();
	realdp *initXptr = initX.ObtainWriteEntireData();
	for (integer i = 0; i < n * p; i++)
		initXptr[i] = X[i];

    // Define the SVPCA problem
    GrassSVPCA Prob(AA, lambda, n, m, p); // DIAGRHESS // LIPSCHITZ
    /*The domain of the problem is a Stiefel manifold*/
    Prob.SetDomain(&Domain);
//    Domain.CheckParams();
    

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
