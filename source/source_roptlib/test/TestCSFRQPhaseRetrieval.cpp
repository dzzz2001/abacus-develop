#include "test/TestCSFRQPhaseRetrieval.h"

#ifdef ROPTLIB_WITH_FFTW

using namespace ROPTLIB;

void testCSFRQPhaseRetrieval(void)
{
    integer n1 = 16, n2 = 16, r = 1, l = 8;
	integer n = n1 * n2, m = n * l;
    realdp kappa = 0; //--- 1e-8;

	CSymFixedRankQ Domain(n, r);
    Vector InitialX = Domain.RandominManifold();
    Domain.ChooseParamsSet1();
//	Domain.CheckParams();
//	Domain.CheckIntrExtr(InitialX);
	//Domain.CheckRetraction(&InitialX);
	//Domain.CheckVecTranDiffRetAdjoint(&InitialX);
	//Domain.CheckVecTranDiffRet(&InitialX, false);
	//Domain.CheckIsometryofVectorTransport(&InitialX);

	//Domain.CheckLockingCondition(&InitialX);
	//Domain.CheckIsometryofInvVectorTransport(&InitialX);
	//Domain.CheckVecTranComposeInverseVecTran(&InitialX);
	//Domain.CheckTranHInvTran(&InitialX);
//	return;

	//InitialX.Print("initialX:");

	// Generate the matrices in the Low rank approximation problem.
//    Vector b(m); b.RandGaussian();
    Vector masks(n, l, "complex"); masks.RandGaussian();
    
    Vector xtrue(n, r, "complex"); xtrue.RandGaussian();

    Vector ZY(m, r, "complex");
    realdp sqn = sqrt(static_cast<realdp> (n1 * n2));
    Vector tmpv;

    for(integer i = 0; i < l; i++)
    {
        for(integer j = 0; j < r; j++)
        {
            tmpv = (masks.GetSubmatrix(0, n - 1, i, i).GetHadamardProduct(xtrue.GetSubmatrix(0, n - 1, j, j))) / sqn;
            ZY.SubmatrixAssignment(i * n, (i + 1) * n - 1, j, j, tmpv.Reshape(n1, n2).GetFFT2D(FFTW_FORWARD).Reshape(n, 1));
        }
    }
    Vector b = ZY.GetTranspose().GetColNormsSquare().GetTranspose();
    
    
    
//    {
//        Vector soln(InitialX);
//        integer maxiter = 1000;
//        realdp time, gfgf0;
//        integer nfft, iter;
//        WFlow(InitialX.ObtainWritePartialData(), b.ObtainWritePartialData(), masks.ObtainWritePartialData(), n1, n2, l, r, maxiter, soln.ObtainWriteEntireData(), time, nfft, iter, gfgf0);
//    }
//    return;
    
    CSFRQPhaseRetrieval Prob(b, masks, kappa, n1, n2, l, r);
	Prob.SetDomain(&Domain);

    Domain.CheckParams();
//	std::cout << "f:" << Prob.f(InitialX) << std::endl;//----
//    Prob.EucGrad(InitialX).Print("EucGrad:");//---

	//Vector *gf = Domain.GetEMPTYINTR()->ConstructEmpty();
	//Prob.Grad(&InitialX, gf);
	//delete gf;
//    Prob.SetNumGradHess(true);


	realdp time, gfgf0;
	integer nfft, iter;
	Vector soln(InitialX);
	WFlow(InitialX.ObtainWritePartialData(), b.ObtainWritePartialData(), masks.ObtainWritePartialData(), n1, n2, l, r, 1000, soln.ObtainWritePartialData(), time, nfft, iter, gfgf0);

//	Prob.CheckGradHessian(InitialX);
//    return;
//    RTRNewton *RSDsolver = new RTRNewton(&Prob, &InitialX);
    LRBFGS *RSDsolver = new LRBFGS(&Prob, &InitialX);
	//->LineSearch_LS = ARMIJO;
	//RSDsolver->LS_beta = 0.01;
	//RSDsolver->RCGmethod = DAI_YUAN;
    RSDsolver->Verbose = ITERRESULT;
	RSDsolver->OutputGap = 1;
	RSDsolver->Max_Iteration = 200;
	//RSDsolver->CheckParams();
	RSDsolver->Accuracy = 1e-6;
//	RSDsolver->Finalstepsize = 1;
	RSDsolver->Tolerance = 1e-6;
    RSDsolver->CheckParams();
	RSDsolver->Run();
	if (RSDsolver->Getnormgfgf0() < 1e-6)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");
//    RSDsolver->GetXopt().Print("xopt:");//---
//	Prob.CheckGradHessian(RSDsolver->GetXopt());//--

	delete RSDsolver;
};

void WFegf(realdp *x, realdp *b, realdp *masks, realdp *egf, integer n1, integer n2, integer l, integer r)
{
    integer m = n1 * n2 * l;
    realdp *cD = new realdp[m];
    realdp *ZY = new realdp[2 * m * r];
    realdp sqn = sqrt(static_cast<realdp> (n1 * n2));

    for (integer i = 0; i < m; i++)
        cD[i] = 0;

    realdp *sZqi = new realdp[n1 * n2];

    for (integer i = 0; i < l; i++)
    {
        for (integer j = 0; j < n1 * n2; j++)
            sZqi[j] = 0;

        for (integer j = 0; j < r; j++)
        {
            for (integer k = 0; k < n1 * n2; k++)
            {
                ZY[i * 2 * n1 * n2 + 2 * k + j * 2 * m] = (x[2 * k + j * 2 * n1 * n2] * masks[2 * k + i * 2 * n1 * n2] - x[2 * k + 1 + j * 2 * n1 * n2] * masks[2 * k + 1 + i * 2 * n1 * n2]) / sqn;
                ZY[i * 2 * n1 * n2 + 2 * k + 1 + j * 2 * m] = (x[2 * k + 1 + j * 2 * n1 * n2] * masks[2 * k + i * 2 * n1 * n2] + x[2 * k + j * 2 * n1 * n2] * masks[2 * k + 1 + i * 2 * n1 * n2]) / sqn;
            }

            fftw_plan p = fftw_plan_dft_2d(n1, n2, (fftw_complex *)(ZY + i * 2 * n1 * n2 + j * 2 * m), (fftw_complex *)(ZY + i * 2 * n1 * n2 + j * 2 * m), FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(p);

            for (integer k = 0; k < n1 * n2; k++)
            {
                sZqi[k] = sZqi[k] + ZY[i * 2 * n1 * n2 + 2 * k + j * 2 * m] * ZY[i * 2 * n1 * n2 + 2 * k + j * 2 * m] + ZY[i * 2 * n1 * n2 + 2 * k + 1 + j * 2 * m] * ZY[i * 2 * n1 * n2 + 2 * k + 1 + j * 2 * m];
            }
        }

        for (integer j = 0; j < n1 * n2; j++)
        {
            cD[j + i * n1 * n2] = sZqi[j] - b[j + i * n1 * n2];
        }
    }
    delete[] sZqi;

    realdp *DZY = new realdp[2 * m * r + 2 * n1 * n2 * r];
    realdp *tmp = DZY + 2 * m * r;
    for (integer i = 0; i < r; i++)
    {
        for (integer j = 0; j < m; j++)
        {
            DZY[2 * j + i * 2 * m] = cD[j] * ZY[2 * j + i * 2 * m];
            DZY[2 * j + 1 + i * 2 * m] = cD[j] * ZY[2 * j + 1 + i * 2 * m];
        }
    }
    delete[] cD;
    delete[] ZY;

    for (integer i = 0; i < 2 * n1 * n2 * r; i++)
        egf[i] = 0;

    integer length = 2 * n1 * n2 * r;
    for (integer i = 0; i < l; i++)
    {
        for (integer j = 0; j < r; j++)
        {
            fftw_plan p = fftw_plan_dft_2d(n1, n2, (fftw_complex *)(DZY + i * 2 * n1 * n2 + j * 2 * m), (fftw_complex *)(DZY + i * 2 * n1 * n2 + j * 2 * m), FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(p);
            
            for (integer k = 0; k < n1 * n2; k++)
            {
                realdp realv = (DZY[2 * k + i * 2 * n1 * n2 + j * 2 * m] * masks[2 * k + i * 2 * n1 * n2] + DZY[2 * k + 1 + i * 2 * n1 * n2 + j * 2 * m] * masks[2 * k + 1 + i * 2 * n1 * n2]) / sqn;
                realdp imagv = (-DZY[2 * k + i * 2 * n1 * n2 + j * 2 * m] * masks[2 * k + 1 + i * 2 * n1 * n2] + DZY[2 * k + 1 + i * 2 * n1 * n2 + j * 2 * m] * masks[2 * k + i * 2 * n1 * n2]) / sqn;
                tmp[2 * k + j * n1 * n2 * 2] = realv;
                tmp[2 * k + 1 + j * n1 * n2 * 2] = imagv;
            }
        }
        axpy_(&length, &GLOBAL::DONE, tmp, &GLOBAL::IONE, egf, &GLOBAL::IONE);
    }
    delete[] DZY;

    realdp scalar = 4.0 / dot_(&m, b, &GLOBAL::IONE, b, &GLOBAL::IONE);
    scal_(&length, &scalar, egf, &GLOBAL::IONE);
};

realdp tol = 1e-6;

realdp k0;
realdp mumax;
realdp mu0;

void WFlow(realdp *initX, realdp *b, realdp *masks, integer n1, integer n2, integer l, integer r, integer maxiter,
           realdp *outsoln, realdp &outtime, integer &outnfft, integer &outiter, realdp &gfgf0)
{
    integer n = n1 * n2; //, m = n * l;
    realdp *egf = new realdp[2 * n * r];
    WFegf(initX, b, masks, egf, n1, n2, l, r);
    integer length = 2 * n * r;
    realdp ngf = sqrt(dot_(&length, egf, &GLOBAL::IONE, egf, &GLOBAL::IONE)), ngf0 = ngf;
    
    unsigned long starttime = getTickCount();
    integer iter = 0;
    realdp deno = dot_(&length, initX, &GLOBAL::IONE, initX, &GLOBAL::IONE);
    realdp stepsize = 0.05;

    copy_(&length, initX, &GLOBAL::IONE, outsoln, &GLOBAL::IONE);
    while (ngf / ngf0 > tol && iter < maxiter)
    {
        /* this stepsize works well for n = 2 ^ 2 to 256 ^ 2*/
        if (l == 6)
        {
            if (n == 8 * 8)
            {
                stepsize = ((1.0 - std::exp(-(1.0 + iter) / 330.0) < 0.13) ? (1.0 - std::exp(-(1.0 + iter) / 330.0)) : 0.13) * 5 / deno;
            }
            else
            if (n == 16 * 16)
            {
                stepsize = ((1.0 - std::exp(-(1.0 + iter) / 330.0) < 0.15) ? (1.0 - std::exp(-(1.0 + iter) / 330.0)) : 0.15) * 20 / deno;
            }
            else
            if (n == 32 * 32)
            {
                stepsize = ((1.0 - std::exp(-(1.0 + iter) / 330.0) < 0.17) ? (1.0 - std::exp(-(1.0 + iter) / 330.0)) : 0.17) * 50 / deno;
            }
        }
        
        if (l == 8)
        {
            stepsize = ((1.0 - std::exp(-(1.0 + iter) / k0) < mumax) ? (1.0 - std::exp(-(1.0 + iter) / k0)) : mumax) * mu0 / deno;
//            stepsize = ((1.0 - std::exp(-(1.0 + iter) / 330.0) < 0.010 * std::log(static_cast<realdp> (n)) - 0.015) ? (1.0 - std::exp(-(1.0 + iter) / 330.0)) : 0.010 * std::log(static_cast<realdp> (n)) - 0.015) * (std::exp(2.00 * pow(static_cast<realdp> (n), 0.125) - 1.7)) / deno;
        }

        if (l == 20)
        {
            stepsize = ((1.0 - std::exp(-(1.0 + iter) / 330.0) < 0.036 * std::log(static_cast<realdp> (n)) - 0.015) ? (1.0 - std::exp(-(1.0 + iter) / 330.0)) : 0.036 * std::log(static_cast<realdp> (n)) - 0.015) * (std::exp(2.35 * pow(static_cast<realdp> (n), 0.125) - 1.7)) / deno;
        }
        WFegf(outsoln, b, masks, egf, n1, n2, l, r);
        ngf = sqrt(dot_(&length, egf, &GLOBAL::IONE, egf, &GLOBAL::IONE));
        if ((iter % 100) == 0)
        {
            printf("iter:%d, ngf:%3.2e, ngf/ngf0:%3.2e, stepsize:%3.2e\n", iter, ngf, ngf/ngf0, stepsize);
        }
        realdp scalar = -stepsize;
        axpy_(&length, &scalar, egf, &GLOBAL::IONE, outsoln, &GLOBAL::IONE);
        iter++;
    }
    printf("total iter:%d, ngf:%3.2e, ngf/ngf0:%3.2e, stepsize:%3.2e\n", iter, ngf, ngf / ngf0, stepsize);

    delete[] egf;

    outtime = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
    outnfft = (iter + 1) * 2 * l;
    outiter = iter;
    gfgf0 = ngf / ngf0;
};

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 8)
	{
		mexErrMsgTxt("The number of arguments should be at least eight.\n");
	}
//    std::cout << "h1" << std::endl;//--
	realdp *b, *masks, *X;
	realdp kappa;
    integer HasHHR, ParamSet, isRMethod;
	b = mxGetPr(prhs[0]);
//    std::cout << "h2" << std::endl;//--
    masks = (realdp *) mxGetComplexDoubles(prhs[1]);
//    std::cout << "h3" << std::endl;//--
	X = (realdp *) mxGetComplexDoubles(prhs[2]);
//    std::cout << "h4" << std::endl;//--
    kappa = mxGetScalar(prhs[3]);
//    std::cout << "h5" << std::endl;//--
    ParamSet = mxGetScalar(prhs[4]);
//    std::cout << "h6" << std::endl;//--
    HasHHR = mxGetScalar(prhs[5]);
    isRMethod = mxGetScalar(prhs[6]);
//    std::cout << "h7" << std::endl;//--
	
	/* dimensions of input matrices */
	integer m, n1, n2, l, n, r;
	const size_t *size = mxGetDimensions(prhs[1]); /* masks */
	n1 = size[0];
	n2 = size[1];
	l = size[2];
	m = mxGetM(prhs[0]);
	n = mxGetM(prhs[2]);
	r = mxGetN(prhs[2]);
	if (n != n1 * n2)
	{
		mexErrMsgTxt("The size of masks or the size of initX is not correct.\n");
	}
	if (m != n * l)
	{
		mexErrMsgTxt("The size of b or the size of masks is not correct.\n");
	}

	genrandseed(0);
	CheckMemoryDeleted = new std::map<integer *, integer>;

    CSymFixedRankQ Domain(n, r);
    if (ParamSet == 1)
        Domain.ChooseParamsSet1();
    else if (ParamSet == 2)
        Domain.ChooseParamsSet2();
    else if (ParamSet == 3)
        Domain.ChooseParamsSet3();
    else
        Domain.ChooseParamsSet4();
    
    Domain.SetHasHHR(HasHHR != 0);
    
    Variable initX = Domain.RandominManifold();
    realdp *initXptr = initX.ObtainWriteEntireData();
    for(integer i = 0; i < initX.Getlength(); i++)
        initXptr[i] = X[i];
    
    Vector mmasks(n, l, "complex");
    realdp *mmasksptr = mmasks.ObtainWriteEntireData();
    for(integer i = 0; i < mmasks.Getlength(); i++)
        mmasksptr[i] = masks[i];
    
    Vector bb(m);
    realdp *bbptr = bb.ObtainWriteEntireData();
    for(integer i = 0; i < bb.Getlength(); i++)
        bbptr[i] = b[i];
    
    CSFRQPhaseRetrieval Prob(bb, mmasks, kappa, n1, n2, l, r);
    Prob.SetDomain(&Domain);
    Prob.SetNumGradHess(false);
    
    if(isRMethod)
        ParseSolverParamsAndOptimizing(prhs[7], &Prob, &initX, plhs);
    else
    {
        plhs[0] = mxCreateDoubleMatrix(n1 * n2, 1, mxCOMPLEX);
        realdp *soln = (realdp *) mxGetComplexDoubles(plhs[0]);
        realdp time, gfgf0;
        integer nfft, iter;

        k0 = mxGetScalar(prhs[8]);
        mumax = mxGetScalar(prhs[9]);
        mu0 = mxGetScalar(prhs[10]);
        integer maxiter = mxGetScalar(prhs[11]);
        
        WFlow(initX.ObtainWritePartialData(), bb.ObtainWritePartialData(), mmasks.ObtainWritePartialData(), n1, n2, l, r, maxiter, soln, time, nfft, iter, gfgf0);
        plhs[1] = mxCreateDoubleScalar(static_cast<double> (time));
        plhs[2] = mxCreateDoubleScalar(static_cast<double> (nfft));
        plhs[3] = mxCreateDoubleScalar(static_cast<double> (iter));
        plhs[4] = mxCreateDoubleScalar(static_cast<double> (gfgf0));
        Vector xopt(initX);
        realdp *xoptptr = xopt.ObtainWriteEntireData();
        for(integer i = 0; i < xopt.Getlength(); i++)
            xoptptr[i] = soln[i];
        plhs[5] = mxCreateDoubleScalar(static_cast<double> (Prob.f(xopt)));
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
#endif
