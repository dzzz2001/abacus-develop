
#include "test/TestObliqueSPCA.h"

using namespace ROPTLIB;

void testObliqueSPCA(void)
{
	// size of the Stiefel manifold
    integer n = 256, m = 20, p = 4;
//    integer n = 128, m = 20, p = 4;
    realdp lambda = 2;

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
    integer numoftypes = 1, numofmani = p;
    Sphere mani(n);
    mani.ChooseParamsSet3();
    ProductManifold Domain(numoftypes, &mani, numofmani);
    Domain.SetIsIntrApproach(false);
    
    Variable ObliqueX = Domain.RandominManifold();
//    realdp *X0ptr = ObliqueX.GetElement(1).ObtainWritePartialData();
//    ObliqueX.Print("X:");//---
//    return;
    
    // Define the SPCA problem
    ObliqueSPCA Prob(A, lambda, n, m, p); // DIAGRHESS // LIPSCHITZ
    /*The domain of the problem is a Stiefel manifold*/
    Prob.SetDomain(&Domain);
    Domain.CheckParams();
    
//    IARPG *IARPGsolver = new IARPG(&Prob, &ObliqueX);
//    IARPGsolver->Max_Iteration = 1000;
//    IARPGsolver->Verbose = ITERRESULT;
//    IARPGsolver->Variant = LSPG_REGULAR; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
//    IARPGsolver->CheckParams();
//    IARPGsolver->Run();
//
////    IARPGsolver->GetXopt().Print("Xopt:");
//
//    delete IARPGsolver;
    
    IRPG *IRPGsolver = new IRPG(&Prob, &ObliqueX);
    IRPGsolver->Max_Iteration = 5000;
    IRPGsolver->Verbose = ITERRESULT;
    IRPGsolver->Variant = LSPG_BB; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ; LSPG_BB
    IRPGsolver->CheckParams();
    IRPGsolver->Run();
    delete IRPGsolver;
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
{ /* (A, lambda, Xinitial, fIRPG, SolverParams), A is m by n matrix, Xinitial is n by p matrix */
	if(nrhs < 5)
	{
		mexErrMsgTxt("The number of arguments should be at least five.\n");
	}
	realdp *A, *X;
    realdp lambda, fIRPG;
	A = mxGetPr(prhs[0]);
    lambda = static_cast<realdp> (mxGetScalar(prhs[1]));
    fIRPG = static_cast<realdp> (mxGetScalar(prhs[3]));
	X = mxGetPr(prhs[2]);
	/* dimensions of input matrices */
	integer m, n, p, HasHHR;
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
    p = mxGetN(prhs[2]);

	/*Check the correctness of the inputs*/
	if(mxGetM(prhs[2]) != n)
	{
		mexErrMsgTxt("The size of matrix Xinit is not correct.\n");
	}
	HasHHR = 0;

	genrandseed(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;

    Vector AA(m, n);
    realdp *AAptr = AA.ObtainWriteEntireData();
    for(integer i = 0; i < m * n; i++)
        AAptr[i] = A[i];
    
    
    // Define the manifold
    integer numoftypes = 1, numofmani = p;
    Sphere mani(n);
    mani.ChooseParamsSet3();
    ProductManifold Domain(numoftypes, &mani, numofmani);
    Domain.SetIsIntrApproach(false);
    
//    // Define the manifold
//    Stiefel Domain(n, p);
//    Domain.ChooseParamsSet4();

	// Set the initial iterate
	Variable initX = Domain.RandominManifold();
	realdp *initXptr = initX.ObtainWriteEntireData();
	for (integer i = 0; i < n * p; i++)
		initXptr[i] = X[i];

    // Define the SPCA problem
    ObliqueSPCA Prob(AA, lambda, n, m, p); // DIAGRHESS // LIPSCHITZ
    /*The domain of the problem is a Stiefel manifold*/
    Prob.SetDomain(&Domain);

    // Call the function defined in DriverMexProb.h
    ParseSolverParamsAndOptimizing(prhs[4], &Prob, &initX, plhs);

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
