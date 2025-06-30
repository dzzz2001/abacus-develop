
#include "test/TestStieSPCA.h"

using namespace ROPTLIB;

Vector &invHV(const Vector &B, realdp alpha, const Vector &V, Vector *result)
{
    integer p = B.Getcol(), r = V.Getcol();
    Vector BtV(p, r), BtB(p, p), Ip(p, p); Ip.SetToIdentity();
    BtV.AlphaABaddBetaThis(1, B, GLOBAL::T, V, GLOBAL::N, 0);
    BtB.AlphaABaddBetaThis(1, B, GLOBAL::T, B, GLOBAL::N, 0);
    Vector tmp = alpha * Ip - 2.0 * BtB;
    Vector tmp2 = tmp % BtV;
    *result = V;
    result->AlphaABaddBetaThis(2, B, GLOBAL::N, tmp2, GLOBAL::N, 1);
    result->ScalarTimesThis(1/alpha);
    return *result;
};

Vector SOCforSPCA(Vector B, realdp mu, Vector initX, realdp fval, integer maxiter, realdp tol, realdp * fvs, integer &lengthfvs, realdp &ComTime)
{
//    B.Print("B:");//---
//    std::cout << "mu:" << mu << std::endl;//---
//    initX.Print("intX:");//---

    unsigned long starttime = getTickCount();
    
    B.SVDDecom();
    Vector S = B.Field("_S");
    const realdp *Sptr = S.ObtainReadData();
//    S.Print("S:");//---
    integer n = initX.Getrow(), p = initX.Getcol();
    
    realdp beta = 2.0 * Sptr[0] * Sptr[0];
    realdp tmpv = 0, normXmP, normQmP, normQ, normX, normP, fv;
    integer iter = 0;
    Vector X = initX, Q = initX, Gamma(n, p), Lambda(n, p), P = initX;
    Gamma.SetToZeros();
    Lambda.SetToZeros();
    
    Vector tmp1(n, p), tmp2(n, p), tmp3(n, p);

    tmp2 = B.GetTranspose() * P; tmp3 = B.GetTranspose() * X;
    fv = - tmp2.DotProduct(tmp3) + mu * X.Onenorm();
    fvs[0] = fv;
    
    /*parameters*/
//    integer maxiter = 500;
//    realdp tol = 1e-4;
    
    for(iter = 0; iter < maxiter; iter++)
    {
        /*update P*/
        tmp1 = beta * (X - Gamma + Q - Lambda);
//        std::cout << "iter:" << iter << std::endl;//---
//        tmp1.Print("LZ:");//---
        invHV(B, 2.0 * beta, tmp1, &P);
//        Vector In(n, n); In.SetToIdentity();
//        ((-2.0 * B * B.GetTranspose() + 2.0 * beta * In) * P - tmp1).Print("hhh:");//---
//        P.Print("X:");//---
        
        /*Update Q*/
        realdp *Qptr = Q.ObtainWriteEntireData();
        const realdp *Lambdaptr = Lambda.ObtainReadData();
        const realdp *Pptr = P.ObtainReadData();
        for(integer i = 0; i < n * p; i++)
        {
            tmpv = std::fabs(Pptr[i] + Lambdaptr[i]) - mu / beta;
            Qptr[i] = (tmpv < 0) ? 0 : tmpv * (Pptr[i] + Lambdaptr[i] > 0 ? 1.0 : -1.0);
        }
//        Q.Print("Q:");//---
        
        /*Update X*/
        X = P + Gamma;
        X.SVDDecom();
        X = X.Field("_U") * X.Field("_Vt");
        
        Gamma = Gamma + P - X;
        Lambda = Lambda + P - Q;

        tmp2 = B.GetTranspose() * P; tmp3 = B.GetTranspose() * X;
        fv = - tmp2.DotProduct(tmp3) + mu * X.Onenorm();
        fvs[iter + 1] = fv;
        lengthfvs = iter + 2;
        if(iter > 1)
        {
            normXmP = (X - P).Fnorm();
            normQmP = (Q - P).Fnorm();
            normX = p;
            normQ = Q.Fnorm();
            normP = P.Fnorm();
            if(normQmP / ( (normP < normQ) ? (normQ < 1 ? 1 : normQ ) : (normP < 1 ? 1 : normP)) + normXmP / ( (normP < normX) ? (normQ < 1 ? 1 : normX ) : (normP < 1 ? 1 : normP)) < tol)
            {
                if(iter % 1000 == 0)
                    printf("SC satisfied: %d, %g\n", iter, fv);
                if(fv <= fval + 1e-7)
                    break;
            }
        }
        if(iter % 1000 == 0)
            printf("iter:%d, f:%.12e\n", iter, fv);
    }
    ComTime = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
    
    printf("Final: iter:%d, f:%.12e, Time:%f\n", iter, fv, ComTime);
    
    return X;
};

void testStieSPCA(void)
{
	// size of the Stiefel manifold
//    integer n = 5, m = 4, p = 3;
    integer n = 1000, m = 20, p = 5;
    realdp lambda = 2;

	// Generate the matrices in the Brockett problem.
    Vector B(m, n);
    realdp *Bptr = B.ObtainWriteEntireData();
	/*B is an n by n matrix*/
	for (integer i = 0; i < n * m; i++)
	{
        Bptr[i] = genrandnormal();
	}
    for(integer i = 0; i < n; i++)
    {
        realdp s = 0;
        for(integer j = 0; j < m; j++)
        {
            s += Bptr[j + i * m];
        }
        s /= m;
        for(integer j = 0; j < m; j++)
            Bptr[j + i * m] -= s;
        s = 0;
        for(integer j = 0; j < m; j++)
        {
            s += Bptr[j + i * m] * Bptr[j + i * m];
        }
        s = std::sqrt(s);
        for(integer j = 0; j < m; j++)
            Bptr[j + i * m] /= s;
    }
    

//    Variable StieX(n, p);
//
//    StieX.RandInManifold();
//    ForDebug::Print("B", B, m, n);//---
//    StieX.Print("StieX:");//---
    // Define the manifold
    Stiefel Domain(n, p);
//    Domain.ChooseParamsSet5();
    Domain.ChooseParamsSet4();
    Variable StieX = Domain.RandominManifold();
    
//    Domain.CheckVecTranDiffRet(StieX, false);
//    Domain.CheckInverseVecTranDiffRet(StieX);
//    Domain.CheckInverseVecTranDiffRetAdjoint(StieX);
//
//    return;
//    /*test SOC*/
//    integer maxiter = 500, lengthfvs;
//    realdp ComTime;
//    Vector fvs(maxiter + 1); fvs.SetToZeros();
//    realdp *fvsptr = fvs.ObtainWriteEntireData();
//    SOCforSPCA(B.GetTranspose(), lambda, StieX, -100, maxiter, 1e-4, fvsptr, lengthfvs, ComTime);
////    fvs.Print("fvs:");
//    return;
    
    integer lengthW = 1;
    // Define the SPCA problem
    StieSPCA Prob(B, lambda, n, m, p, lengthW); // DIAGRHESS // LIPSCHITZ
    /*The domain of the problem is a Stiefel manifold*/
    Prob.SetDomain(&Domain);
    Domain.CheckParams();
    
//    ARPG *ARPGsolver = new ARPG(&Prob, &StieX);
//    ARPGsolver->Max_Iteration = 5000;
//    ARPGsolver->Variant = REGULAR; //-- REGULAR; //--ADALIPSCHITZ;
//    ARPGsolver->CheckParams();
//    ARPGsolver->Run();
//    delete ARPGsolver;
    
    IARPG *IARPGsolver = new IARPG(&Prob, &StieX);
//    IARPGsolver->Max_Iteration = 500;
//    IARPGsolver->Verbose = ITERRESULT;
//    IARPGsolver->Variant = LSPG_REGULAR; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    IARPGsolver->CheckParams();
    IARPGsolver->Run();
    delete IARPGsolver;
    
//    IRPG *IRPGsolver = new IRPG(&Prob, &StieX);
////    IRPGsolver->Max_Iteration = 300;
////    IRPGsolver->Verbose = ITERRESULT;
////    IRPGsolver->OutputGap = 20;
////    IRPGsolver->ProxMapType = LSPG_LOCAL; //-- LSPG_UNILIMIT; LSPG_GLOBAL LSPG_LOCAL
////    IRPGsolver->Variant = LSPG_ADALIPSCHITZ; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
//    IRPGsolver->CheckParams();
//    IRPGsolver->Run();
//    delete IRPGsolver;
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
{ /* (A, lambda, Xinitial, method, fIRPG, SolverParams) */
	if(nrhs < 6)
	{
		mexErrMsgTxt("The number of arguments should be at least six.\n");
	}
	realdp *A, *X;
    realdp lambda, fIRPG;
    integer method;
	A = mxGetPr(prhs[0]);
    lambda = static_cast<realdp> (mxGetScalar(prhs[1]));
    method = static_cast<integer> (mxGetScalar(prhs[3]));
    fIRPG = static_cast<realdp> (mxGetScalar(prhs[4]));
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
    Stiefel Domain(n, p);
    Domain.ChooseParamsSet4();

	// Obtain an initial iterate by taking the Q factor of qr decomposition
	Variable initX = Domain.RandominManifold();
	realdp *initXptr = initX.ObtainWriteEntireData();
	for (integer i = 0; i < n * p; i++)
		initXptr[i] = X[i];
    
    if(method == 0) /* use SOC */
    {
        integer maxiter = 50000, lengthfvs;
        realdp *fvs = new realdp[maxiter + 1];
        
        realdp comtime;
        Vector Xopt = SOCforSPCA(AA.GetTranspose(), lambda, initX, fIRPG, maxiter, 1e-5, fvs, lengthfvs, comtime);
        mexProblem::ObtainMxArrayFromElement(plhs[0], &Xopt);
        plhs[1] = mxCreateDoubleMatrix(lengthfvs, 1, mxREAL);
        plhs[2] = mxCreateDoubleScalar(comtime);
        double *mxfvs = mxGetPr(plhs[1]);
        for(integer i = 0; i < lengthfvs; i++)
        {
            mxfvs[i] = fvs[i];
        }
        delete [] fvs;
    }
    else
    {
        mxArray *tmp = mexProblem::GetFieldbyName(prhs[5], 0, "LengthW");
        integer lengthW = static_cast<integer> (mxGetScalar(tmp));
//        std::cout << "lengthW:" << lengthW << std::endl;//----
        // Define the Brockett problem
        StieSPCA Prob(AA, lambda, n, m, p, lengthW); // DIAGRHESS // LIPSCHITZ
        Prob.SetDomain(&Domain);

        Domain.SetHasHHR(HasHHR != 0);
        //Domain.CheckParams();

        // Call the function defined in DriverMexProb.h
        ParseSolverParamsAndOptimizing(prhs[5], &Prob, &initX, plhs);
    }

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
