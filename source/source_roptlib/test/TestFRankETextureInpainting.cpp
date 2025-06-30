
#include "test/TestFRankETextureInpainting.h"

using namespace ROPTLIB;

#ifdef ROPTLIB_WITH_FFTW

void testFRankETextureInpainting(void)
{
//    Vector AA(512, 500), BB(500, 20), CC;
//    AA.RandGaussian();
//    BB.RandGaussian();
//    unsigned long starttime = getTickCount();
//    for(integer i = 0; i < 100; i++)
//    {
//        AA * BB;
//    }
////    AA.SVDDecom();
//    printf("time:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//
//    return;
    
	//	FILE *ttt = nullptr;
	//	freopen_s(&ttt, "./log.txt", "w", stdout);
//    integer m = 8, n = 6, r = 3;
integer m = 200, n = 180, r = 50;
//integer m = 512, n = 500, r = 20;
//    integer m = 10, n = 10, r = 2;
//	integer m = 1000, n = 1000, r = 10;
	FixedRankE Domain(m, n, r);
	Domain.SetHasHHR(false);
    Domain.CheckParams();
    std::cout << "Generating initial iterate" << std::endl;
    Variable InitialX = Domain.RandominManifold();
//    InitialX.Print("X:", false);//---
//	Variable InitialX(m, n, r);
//	InitialX.RandInManifold();
//  Domain.CheckParams();
//  Domain.CheckIntrExtr(&InitialX);
//	Domain.CheckRetraction(InitialX);
//    Domain.CheckVecTranDiffRet(InitialX, true);
//    Domain.CheckLockingCondition(InitialX);
//	Domain.CheckVecTranDiffRetAdjoint(&InitialX);
//	Domain.CheckVecTranDiffRet(&InitialX, false);
//	Domain.CheckIsometryofVectorTransport(&InitialX);
//
//	Domain.CheckLockingCondition(&InitialX);
//	Domain.CheckIsometryofInvVectorTransport(&InitialX);
//	Domain.CheckVecTranComposeInverseVecTran(&InitialX);
//	Domain.CheckTranHInvTran(&InitialX);
    
////    Domain.CheckVecTranDiffRet(InitialX, false);
////    Domain.CheckVecTranDiffRetAdjoint(GrassX);
//    Domain.CheckInverseVecTranDiffRet(InitialX);
//    Domain.CheckInverseVecTranDiffRetAdjoint(InitialX);
//    return;
    
    // Generate the matrices in the matrix completion approximation problem.
    std::cout << "Generating the index set Omega" << std::endl;
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
    
	// Generate the matrices in the Low rank approximation problem.
    std::cout << "Generating low-rank and sparse matrix A" << std::endl;
    Vector A(m, n); //, B(m, n), G(m, r), H(r, n);
    for(integer i = 0; i < r; i++)
    {
        Vector v(m, 1); v.RandGaussian(); v.ScalarTimesThis(1/std::sqrt(m));
        realdp *vptr = v.ObtainWritePartialData();
        for(integer j = 0; j < m; j++)
        {
            if(vptr[j] < 0)
                vptr[j] = 0;
        }
//        v.Print("v:");//---
        for(integer j = (integer ) (static_cast<realdp> (i) / r * n); j < (integer ) (static_cast<realdp> (i+1) / r * n); j++)
        {
            A.SubmatrixAssignment(0, m - 1, j, j, v);
        }
    }
//    A.Print("A");
//    return;
    
//    G.RandGaussian(); H.RandGaussian(); A = G * H; B.RandGaussian();
//    for(integer i = 0; i < m * n; i++)
//    {
//        if(B.ObtainReadData()[i] < 0)
//            A.ObtainWriteEntireData()[i] = 0;
//    }
    
//    A.Print("A:");//---
    
    Vector solntrue = A;
    std::cout << "Generating a transformation of A" << std::endl;
    integer type = 2;
    if(type == 1)
        A = A.GetInvHaarFWT();
    else
        A = A.GetDCST2D(FFTW_REDFT10);
    
//    A.Print("DCT A:");//---
    
    realdp lambda = 0.001;
    integer lengthW = 1;
    std::cout << "Creating the problem" << std::endl;
    FRankETextureInpainting Prob(ir, jc, jcc, nz, A, lambda, m, n, r, lengthW, type);
	Prob.SetDomain(&Domain);

//    Vector xoptLADM = LADM(ir, jc, jcc, nz, A, 0.001, m, n, r, type, InitialX);
    std::cout << std::endl;
    std::cout << std::endl;
//    return;
//	Prob.CheckGradHessian(InitialX);

//        Domain.CheckVecTranDiffRet(InitialX, false);
//        Domain.CheckInverseVecTranDiffRet(InitialX);
//        Domain.CheckInverseVecTranDiffRetAdjoint(InitialX);
//        return;
//    return;
    
    IRPG *IARPGsolver = new IRPG(&Prob, &InitialX);
    IARPGsolver->Max_Iteration = 300;
    IARPGsolver->Variant = LSPG_ADALIPSCHITZ; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ; LSPG_BB
    IARPGsolver->OutputGap = 1;
    IARPGsolver->Minstepsize = 1e-6;
    IARPGsolver->SMtol = 1e-2;
    IARPGsolver->ProxMapType = LSPG_GLOBAL; //-- LSPG_UNILIMIT; LSPG_GLOBAL
    IARPGsolver->CheckParams();
    IARPGsolver->Run();
    delete IARPGsolver;
    
    IRPG *IARPGsolver2 = new IRPG(&Prob, &InitialX);
    IARPGsolver2->Max_Iteration = 100;
    IARPGsolver2->Variant = LSPG_ADALIPSCHITZ; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ; LSPG_BB
    IARPGsolver2->OutputGap = 1;
    IARPGsolver2->Minstepsize = 1e-6;
    IARPGsolver2->SMtol = 1e-2;
//    IARPGsolver2->Tolerance = 1e-6;
    IARPGsolver2->ProxMapType = LSPG_UNILIMIT; //-- LSPG_UNILIMIT; LSPG_GLOBAL
    IARPGsolver2->CheckParams();
    IARPGsolver2->Run();
    delete IARPGsolver2;
    
    
    delete [] ir;
    delete [] jcc;
};


Vector LADM(unsigned long *inir, unsigned long *injc, unsigned long *injcc, unsigned long innzmax, Vector inD, realdp inlambda, integer inm, integer inn, integer inr, 
        realdp mu, realdp rho, realdp eta, integer type, Vector W, integer *outIter, realdp *outComTime)
{
    Vector Y1(inm, inn), Y2(inm, inn), A(inm, inn);
    Y1.RandGaussian(); Y2.RandGaussian();
//     realdp mu = 0.001, rho = 1.1, eta = 3;
    realdp err1 = 0, err2 = 0, diffW = 0;
    integer iter = 0, NumRank = 0;;
    unsigned long starttime = getTickCount();
//     printf("mu:%f\n", mu);//---
//     std::cout << "mu...:" << mu << std::endl;//---
    while(iter < 1000)
    {
        /* update A */
        Vector tmp = W - Y1 / mu;
        tmp.SVDDecom();
        Vector U = tmp.Field("_U"), S = tmp.Field("_S"), Vt = tmp.Field("_Vt");
        realdp *Sptr = S.ObtainWritePartialData();
        for(integer i = 0; i < S.Getlength(); i++)
        {
            Sptr[i] = (fabs(Sptr[i]) < 1.0 / mu) ? 0 : ( (Sptr[i] < 0)? Sptr[i] + 1.0 / mu : Sptr[i] - 1.0 / mu );
        }
        
        for(integer i = 0; i < S.Getlength(); i++)
        {
            if(Sptr[i] == 0)
            {
                NumRank = i;
                break;
            }
        }
//         S.Print("S:");//---
        A = S.GetDiagTimesM(U, GLOBAL::R) * Vt;
        
        /* update W */
        Vector B1WB2t = ((type == 1) ? W.GetInvHaarFWT() : W.GetDCST2D(FFTW_REDFT10));
        Vector tmp1(inm, inn); tmp1.SetToZeros();
        const realdp *B1WB2tptr = B1WB2t.ObtainReadData();
        const realdp *inDptr = inD.ObtainReadData();
        const realdp *Y2ptr = Y2.ObtainReadData();
        realdp *tmp1ptr = tmp1.ObtainWriteEntireData();
        for(integer i = 0; i < innzmax; i++)
        {
            tmp1ptr[inir[i] + injc[i] * inm] = B1WB2tptr[inir[i] + injc[i] * inm] - inDptr[inir[i] + injc[i] * inm] + Y2ptr[inir[i] + injc[i] * inm] / mu;
        }
        Vector tmp2 = ((type == 1) ? tmp1.GetHaarFWT() : tmp1.GetDCST2D(FFTW_REDFT01));
        Vector tmp3(inm, inn);
        const realdp *tmp2ptr = tmp2.ObtainReadData();
        const realdp *Wptr = W.ObtainReadData();
        const realdp *Aptr = A.ObtainReadData();
        const realdp *Y1ptr = Y1.ObtainReadData();
        realdp *tmp3ptr = tmp3.ObtainWriteEntireData();
        for(integer i = 0; i < inm * inn; i++)
        {
            tmp3ptr[i] = Wptr[i] - (tmp2ptr[i] + Wptr[i] - Aptr[i] - Y1ptr[i] / mu) / eta;
            tmp3ptr[i] = (fabs(tmp3ptr[i]) < inlambda / mu / eta ) ? 0 : ( (tmp3ptr[i] < 0)? tmp3ptr[i] + inlambda / mu / eta : tmp3ptr[i] - inlambda / mu / eta );
        }
        diffW = (W - tmp3).Fnorm() / W.Fnorm();
        W = tmp3;
        
        /* update Y1 */
        Y1 = Y1 + mu * (A - W);
        
        err1 = (A-W).Fnorm() / A.Fnorm();
        
        /* update Y2 */
        Vector tmp4 = ((type == 1) ? W.GetInvHaarFWT() : W.GetDCST2D(FFTW_REDFT10));
        Vector tmp5(inm, inn); tmp5.SetToZeros();
        Y2ptr = Y2.ObtainReadData();
        const realdp *tmp4ptr = tmp4.ObtainReadData();
        realdp *tmp5ptr = tmp5.ObtainWriteEntireData();
        for(integer i = 0; i < innzmax; i++)
        {
            tmp5ptr[inir[i] + injc[i] * inm] = (tmp4ptr[inir[i] + injc[i] * inm] - inDptr[inir[i] + injc[i] * inm]);
        }
        err2 = tmp5.Fnorm() / inD.Fnorm();
        Y2 = Y2 + mu * tmp5;
        
        mu *= rho;

        if(iter % 10 == 0)
            printf("iter:%d, norm(A-W)/norm(A):%e, norm(P(B1 W B2t - D)) / norm(D):%e, diffW:%e, NumRank:%d, \n", iter, err1, err2, diffW, NumRank);
        iter++;
        
        if(err1 + err2 < 1e-3 && diffW < 1e-5)
        {
            break;
        }
//        break;//---
    }
    *outComTime = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
    printf("time:%f\n", *outComTime);
    *outIter = iter;
    W = A;
    return W;
};

#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{/*input: A, lambda, initX, type, method, SolverParams*/
    if(nrhs < 9)
    {
        mexErrMsgTxt("The number of arguments should be at least night.\n");
    }

    realdp *A;
    A = mxGetPr(prhs[0]);
    /* dimensions of input matrices */
    integer m, n, HasHHR, nzmax, r, type, method;
    realdp LADMmu, LADMrho, LADMeta;
    size_t *ir, *jc;
    nzmax = mxGetNzmax(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    realdp lambda = static_cast<realdp> (mxGetScalar(prhs[1]));
    type = static_cast<integer> (mxGetScalar(prhs[3]));
    method = static_cast<integer> (mxGetScalar(prhs[4]));
    LADMmu = static_cast<realdp> (mxGetScalar(prhs[6]));
    LADMrho = static_cast<realdp> (mxGetScalar(prhs[7]));
    LADMeta = static_cast<realdp> (mxGetScalar(prhs[8]));
//     std::cout << "LADMmu:" << LADMmu << std::endl;//---
    
    if (!mxIsStruct(prhs[2]))
    {
        mexErrMsgTxt("The third argument, initial iterate, is not a structure.");
    }
    
    Vector InitialX(m, n);
    mexProblem::ObtainElementFromMxArray(&InitialX, prhs[2]);
    r = InitialX.Field("U").Getcol();
    
    // Get the index set and dense matrix AA
    Vector AA(m, n); AA.SetToZeros();
    realdp *AAptr = AA.ObtainWritePartialData();
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
            AAptr[ir[j] + i * m] = A[j];
        }
        injcc[i] = jc[i];
    }
    injcc[n] = jc[n];
    
    genrandseed(0);

    CheckMemoryDeleted = new std::map<integer *, integer>;
    
    if(method == 1) /* Riemannian method */
    {
        integer lengthW = 1;
        // Define the manifold
        FixedRankE Domain(m, n, r);
        HasHHR = 0;
        Domain.SetHasHHR(HasHHR != 0);
        
        FRankETextureInpainting Prob(inir, injc, injcc, nzmax, AA, lambda, m, n, r, lengthW, type);
        Prob.SetDomain(&Domain);
        // Call the function defined in DriverMexProb.h
        ParseSolverParamsAndOptimizing(prhs[5], &Prob, &InitialX, plhs);
    } else
    {
        integer LADMiter = 0;
        realdp ComTimeLADM = 0;
        Vector xoptLADM = LADM(inir, injc, injcc, nzmax, AA, lambda, m, n, r, LADMmu, LADMrho, LADMeta, type, InitialX, &LADMiter, &ComTimeLADM);
        mexProblem::ObtainMxArrayFromElement(plhs[0], &xoptLADM);
        plhs[1] = mxCreateDoubleScalar(static_cast<double> (LADMiter));
        plhs[2] = mxCreateDoubleScalar(static_cast<double> (ComTimeLADM));
    }

    std::map<integer *, integer>::iterator iter = CheckMemoryDeleted->begin();
    for (iter = CheckMemoryDeleted->begin(); iter != CheckMemoryDeleted->end(); iter++)
    {
        if (iter->second != 1)
            printf("Global address: %p, sharedtimes: %d\n", iter->first, iter->second);
    }
    delete CheckMemoryDeleted;
    delete [] inir;
    return;
}

#endif
#endif
