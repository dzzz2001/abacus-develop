#include "test/DriverCpp.h"

using namespace ROPTLIB;

int main(void)
{
//	_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF); /*This can detect the memory leakage for global variables!!*///
//	_CrtSetBreakAlloc(160);
	/*Set the random seed*/
	unsigned tt = (unsigned)time(NULL);
//    tt = 0; /*The following test is only for random seed zero*/
//    tt = 1597890503;
//    tt = 1597913182; /*LRTRSR1 slow, LRBFGS fast Matrix completion*/
//    tt = 1597970251;
//    tt = 1597977860;
//    tt = 1597966262;
//    tt = 1598884430;
    tt = 2;
	std::cout << "seed:" << tt << std::endl;
	genrandseed(tt);
    
//    std::cout << std::numeric_limits<double>::max() << std::endl;//---
//    integer N = 3;
//    SPDManifold spdMani(3);      // a 3 by 3 SPD manifold
//    spdMani.ChooseParamsSet3();  // the  parameter set is set here.
//    ProductManifold prodMani(1, &spdMani, N);
//
//    prodMani.CheckParams();
//
//    Variable prod_x = prodMani.RandominManifold();
//    Variable prod_y = prodMani.RandominManifold();
//    Vector eta_prod_x = prodMani.GetEMPTY();
//    eta_prod_x.RandGaussian();
//    prodMani.ExtrProjection(prod_x, eta_prod_x, &eta_prod_x);
//    /* if the entries in eta_prod_x and prod_x needs be modified, then the following codes can be used:
//    realdp *prod_xptr = prod_x.ObtainWriteEntireData();
//    realdp *eta_prod_xptr = eta_prod_x.ObtainWriteEntireData();
//    for(integer i = 0; i < N; i++)
//    {
//        for(integer j = 0; j < 3; j++)
//        {
//            for(integer k = 0; k < 3; k++)
//            {
//                prod_xptr[i * 9 + j * 3 + k] = ...;
//                eta_prod_xptr[i * 9 + j * 3 + k] = ...;
//            }
//        }
//    }
//    */
//    prodMani.Retraction(prod_x, eta_prod_x, &prod_y);
//    prod_x.Print("prod_x:");
//    eta_prod_x.Print("eta_prod_x:");
//    prod_y.Print("prod_y:");
    
    
//    testElement();
//    testEucQuadratic();
//    testCFRankQ2FBlindDecon2D();
//    testCStieBrockett();
//    testFRankE3FMatCompletion();
//    testFRankESparseApprox();
//    testFRankETextureInpainting();
//    testFRankEWeightApprox();
//    testFRankQ2FMatCompletion();
//    testGrassMatCompletion();
//    testGrassPCA();
//    testGrassRQ();
//    testGrassSVPCA();
//    testPoincareEmbeddings();
//    testProdStieSumBrockett();
//    testObliqueSPCA();
//    testSPDKarcherMean();
//    testSphereSparsestVector();
//    testStieBrockett();
//    testStieSoftICA();
//    testStieSPCA();
//    testSFRQLyapunov();
//    testCSFRQPhaseRetrieval();

//    integer length = 2;
//    realdp *vals = new realdp [2];
//    vals[0] = 1; vals[1] = 1;
//    integer inc = 1;
//    std::cout << ":::" << sdot_(&length, vals, &inc, vals, &inc) << std::endl;//---
    
	testall();

#ifdef _WIN64
#ifdef _DEBUG
	_CrtDumpMemoryLeaks();
#endif
#endif
	return 0;
}

void testall(void)
{ /*The following test is only for random seed zero*/

  /*Set the random seed*/
	unsigned seed = (unsigned)time(NULL);
	seed = 1; /*The following test is only for random seed zero*/

#ifdef ROPTLIB_WITH_FFTW
    {
        genrandseed(seed);
        printf("\n testCSFRQPhaseRetrieval\n");

        integer n1 = 2, n2 = 2, r = 1, l = 2;
        integer n = n1 * n2, m = n * l;
        realdp kappa = 0.000;

        CSymFixedRankQ Domain(n, r);
//        Domain.SetHasHHR(true);
        Vector InitialX = 3 * (Domain.RandominManifold());
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

        CSFRQPhaseRetrieval Prob(b, masks, kappa, n1, n2, l, r);
//        Domain.CheckParams();//---
        Prob.SetDomain(&Domain);

        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRNewton", "RTRSR1", "LRTRSR1"}; /*"RTRSD", */
        testSmoothProblem(&Prob, &InitialX, "testCSFRQPhaseRetrieval", names);
    }

    {
        genrandseed(seed);
        printf("\n testCFRankQ2FBlindDecon2D\n");

        integer n1 = 2, n2 = 2, r = 1, L = n1 * n2;
        CFixedRankQ2F Domain(L, L, r);
//        Domain.SetHasHHR(true);
        Variable InitialX = Domain.RandominManifold();

        Vector y(L, "complex");
        y.RandGaussian();
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

        CFRankQ2FBlindDecon2D Prob(y, sB, sC, n1, n2, r, 0, 1, 1);
        Prob.SetDomain(&Domain);

        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};

        testSmoothProblem(&Prob, &InitialX, "testCFRankQ2FBlindDecon2D", names);

        sB.Setnullptr();
        sC.Setnullptr();
        delete[] B;
        delete[] irB;
    }
#endif
    
    {
        genrandseed(seed);
        printf("\n testFRankE3FMatCompletion\n");

        integer m = 50, n = 50, r = 4;
        // Generate the matrices in the matrix completion approximation problem.
        integer dim = (m + n - r) * r;
        integer nz = 3 * dim;
        nz = (nz > m + n) ? m + n : nz;
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
        Variable InitialX = Domain.RandominManifold();
        
        FRankE3FMatCompletion Prob(ir, jc, jcc, V, nz, m, n, r);
        Prob.SetDomain(&Domain);

        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &InitialX, "testFRankE3FMatCompletion", names, 1);
        delete[] V;
        delete[] ir;
        delete[] jcc;
    }

    {
        genrandseed(seed);
        printf("\n testGrassRQ\n");

        /* size of the Grassmann manifold */
        integer n = 5, p = 2;
        /* Generate the matrices in the Rayleigh Quotient problem. */
        Vector B(n, n);
        B.RandGaussian();
        B = B + B.GetTranspose();

        // Define the manifold
        Grassmann Domain(n, p);
        Variable InitialX = Domain.RandominManifold();

        /* Define the  problem*/
        GrassRQ Prob(B, n, p);
        /*The domain of the problem is a Stiefel manifold*/
        Prob.SetDomain(&Domain);
        Domain.CheckParams();//---

        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RWRBFGS", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &InitialX, "testGrassRQ", names);
    }
    
    {
        genrandseed(seed);
        printf("\n testSFRQLyapunov\n");

        integer n = 5, p = 2, pC = 1;
        integer nzmaxA = n + 2 * (n - 1);
        realdp *A = new realdp[n + 2 * (n - 1)];
        unsigned long *inirA = new unsigned long[2 * nzmaxA + n + 1];
        unsigned long *injcA = inirA + nzmaxA;
        unsigned long *injccA = injcA + nzmaxA;
        injccA[0] = 0;
        A[0] = 2; inirA[0] = 0; injcA[0] = 0;
        A[1] = -1; inirA[1] = 1; injcA[1] = 0;
        injccA[1] = 2;
        for (integer i = 1; i < n - 1; i++)
        {
            A[3 * i - 1] = -1; inirA[3 * i - 1] = i - 1; injcA[3 * i - 1] = i;
            A[3 * i    ] = 2;  inirA[3 * i    ] = i;     injcA[3 * i] = i;
            A[3 * i + 1] = -1; inirA[3 * i + 1] = i + 1; injcA[3 * i + 1] = i;
            injccA[i + 1] = i * 3 + 2;
        }
        A[3 * n - 4] = -1; inirA[3 * n - 4] = n - 2; injcA[3 * n - 4] = n - 1;
        A[3 * n - 3] = 2;  inirA[3 * n - 3] = n - 1; injcA[3 * n - 3] = n - 1;
        injccA[n] = 3 * n - 2;
        
        SparseMatrix sA(n, n, inirA, injcA, injccA, A, nzmaxA);

        integer nzmaxM = n;
        realdp *M = new realdp[n];
        unsigned long *inirM = new unsigned long[2 * nzmaxM + n + 1];
        unsigned long *injcM = inirM + nzmaxM;
        unsigned long *injccM = injcM + nzmaxM;
        injccM[0] = 0;
        for(integer i = 0; i < n; i++)
        {
            M[i] = 1;
            inirM[i] = i;
            injcM[i] = i;
            injccM[i] = i + 1;
        }
        SparseMatrix sM(n, n, inirM, injcM, injccM, M, nzmaxM);
        
        Vector C(n, pC);
        C.RandGaussian();

        SymFixedRankQ Domain(n, p);
        Vector InitialX = Domain.RandominManifold();
        SFRQLyapunov Prob(sA, sM, C, p);
        Prob.SetDomain(&Domain);
//        Domain.SetHasHHR(true);//--
//        Domain.CheckParams();//----
//        Domain.CheckRetraction(InitialX);//---
//        Domain.CheckIntrExtr(InitialX);//---
//        Domain.CheckVecTranDiffRet(InitialX);//---
//        Domain.CheckIsometryofVectorTransport(InitialX);//--

        /* TOCHECKWHY: LRBroydenFamily does not work correctly for this problem. */
        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"}; //--"LRBroydenFamily",
        testSmoothProblem(&Prob, &InitialX, "testSFRQLyapunov", names);
        sA.Setnullptr();
        sM.Setnullptr();
		delete[] inirM;
		delete[] M;
		delete[] inirA;
		delete[] A;
    }
    
    {
        genrandseed(seed);
        printf("\n testFRankQ2FMatCompletion\n");

        integer m = 8, n = 7, r = 2;
        
        // Generate the matrices in the matrix completion approximation problem.
        integer dim = (m + n - r) * r;
        integer nz = 3 * dim;
        nz = (nz > m + n) ? m + n : nz;
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
            jc[i] = static_cast<unsigned long> (tmpforidx[i] / m);
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
        
        
        FixedRankQ2F Domain(m, n, r);
        Variable InitialX = Domain.RandominManifold();
        Vector G(m, r); G.RandGaussian(); G.QRDecom(); InitialX.GetElement(0) = G.Field("_Q");
        Vector H(n, r); H.RandGaussian(); H.QRDecom(); InitialX.GetElement(1) = H.Field("_Q");
        
        FRankQ2FMatCompletion Prob(ir, jc, jcc, V, nz, m, n, r);
        Prob.SetDomain(&Domain);

        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &InitialX, "testFRankQ2FMatCompletion", names);
        delete[] V;
		delete[] ir;
        delete[] jcc;
    }
    
    {
        genrandseed(seed);
        printf("\n testStieBrockett\n");
        // size of the Stiefel manifold
        integer n = 5, p = 3;
        // Generate the matrices in the Brockett problem.
        Vector B(n, n), D(p);
        B.RandGaussian();
        B = B + B.GetTranspose();
        realdp *Dptr = D.ObtainWriteEntireData();
        /*D is a diagonal matrix.*/
        for (integer i = 0; i < p; i++)
            Dptr[i] = static_cast<realdp> (i + 1);
        // Define the manifold
        Stiefel Domain(n, p);
//        Domain.ChooseParamsSet2();
        //Grassmann Domain(n, p);
//        Domain.SetHasHHR(true); /*set whether the manifold uses the idea in [HGA2015, Section 4.3] or not*/
        Variable StieX = Domain.RandominManifold();
        // Define the Brockett problem
        StieBrockett Prob(B, D);
        /*The domain of the problem is a Stiefel manifold*/
        Prob.SetDomain(&Domain);
        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RWRBFGS", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &StieX, "testStieBrockett", names);
    }
    
    {
        genrandseed(seed);
        printf("\n testProdStieSumBrockett\n");
        integer n = 4, p = 2, m = 3, q = 2;
        Vector B1(n, n), B2(n, n), B3(m, m);
        B1.RandGaussian(); B1 = B1 + B1.GetTranspose();
        B2.RandGaussian(); B2 = B2 + B2.GetTranspose();
        B3.RandGaussian(); B3 = B3 + B3.GetTranspose();
        Vector D1(p), D2(p), D3(q);
        realdp *D1ptr = D1.ObtainWriteEntireData();
        realdp *D2ptr = D2.ObtainWriteEntireData();
        realdp *D3ptr = D3.ObtainWriteEntireData();
        for (integer i = 0; i < p; i++)
        {
            D1ptr[i] = static_cast<realdp> (i + 1);
            D2ptr[i] = D1ptr[i];
        }
        for (integer i = 0; i < q; i++)
        {
            D3ptr[i] = static_cast<realdp> (i + 1);
        }
        // number of manifolds in product of manifold
        integer numoftypes = 2; // two kinds of manifolds
        integer numofmani1 = 2; // the first one has two
        integer numofmani2 = 1; // the second one has one
        // Define the Stiefel manifold
        Stiefel mani1(n, p);
        Stiefel mani2(m, q);
        ProductManifold Domain(numoftypes, &mani1, numofmani1, &mani2, numofmani2);
        // Obtain an initial iterate
        Variable ProdX = Domain.RandominManifold();
        // Define the Brockett problem
        ProdStieSumBrockett Prob(B1, D1, B2, D2, B3, D3);
        // Set the domain of the problem to be the Stiefel manifold
        Prob.SetDomain(&Domain);
        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RWRBFGS", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &ProdX, "testProdStieSumBrockett", names);
    }
    
    {
        genrandseed(seed);
        printf("\n testEucQuadratic\n");
        // size of the domain
        integer dim = 10;
        Vector O(dim, dim);
        O.RandGaussian();
        O = O.GetOrth();
        Vector D(dim);
        D.RandUnform();
        D = D + 0.1;
        Vector A = O.GetTranspose() * D.GetDiagTimesM(O);
        // Obtain an initial iterate
        Euclidean EucDomain(dim);
        Variable EucX = EucDomain.RandominManifold();
        // Define the problem
        EucQuadratic Prob(A);
        Prob.SetDomain(&EucDomain);
        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RWRBFGS", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &EucX, "testEucQuadratic", names);
    }
    
    {
        genrandseed(seed);
        printf("\n testFRankEWeightApprox\n");
        integer m = 5, n = 4, r = 2;
        //integer m = 100, n = 15, r = 5;
        FixedRankE Domain(m, n, r);
        Domain.SetHasHHR(true);
        Variable InitialX = Domain.RandominManifold();
        // Generate the matrices in the Low rank approximation problem.
        Vector A(m, n); A.RandGaussian();
        Vector O(m * n, m * n);
        O.RandGaussian();
        O = O.GetOrth();
        Vector D(m * n);
        D.RandUnform();
        D = D + 0.1;
        Vector W = O.GetTranspose() * D.GetDiagTimesM(O);

        FRankEWeightApprox Prob(A, W, m, n, r);
        Prob.SetDomain(&Domain);
        /* Extrinsic approach is used here and transporting linear opeartor has not been done. Therefore, all algorithms
        that need transporting operators are not tested here. */
        std::vector<std::string> names = {"RSD", "RNewton", "RCG", "LRBFGS", "RTRSD", "RTRNewton"};
        testSmoothProblem(&Prob, &InitialX, "testFRankEWeightApprox", names);
    }
    
    {
        genrandseed(seed);
        printf("\n testGrassMatCompletion\n");
//        integer d = 50, N = 1000, r = 5;
        integer d = 10, N = 50, r = 2;
        
        Grassmann Domain(d, r);
        Domain.SetHasHHR(true);
        Domain.SetIsIntrApproach(true);
        Variable GrassX = Domain.RandominManifold();
        // Generate the matrices in the matrix completion approximation problem.
        integer dim = (d + N - r) * r;
        realdp OS = 3;
        integer nz = (OS * dim > d * N) ? d * N : OS * dim;
        integer *ir = new integer[nz * 2];
        integer *jc = ir + nz;
        integer *tmpforidx = new integer[d * N];
        for (integer i = 0; i < d * N; i++)
            tmpforidx[i] = i;
        /*nz number of indices*/
        integer idx = 0, itmp;
        for (integer i = 0; i < nz; i++)
        {
            /*idx is an integer in [0, m - i - 1]*/
            idx = static_cast<integer> ((d * N - i) * genrandreal());
            while (idx >= d * N - i)
                idx = static_cast<integer> ((d * N - i) * genrandreal());
            /*the chosen idx is put at the end of the array*/
            itmp = tmpforidx[d * N - i - 1];
            tmpforidx[d * N - i - 1] = tmpforidx[idx];
            tmpforidx[idx] = itmp;
        }
        for (integer i = 0; i < nz; i++)
        {
            /*tmpforidx[nz - 1 - i]*/
            ir[i] = static_cast<integer> (tmpforidx[nz - 1 - i] / N);
            jc[i] = tmpforidx[nz - 1 - i] - N * ir[i];
        }
        delete[] tmpforidx;
        
        integer dr = d * r, nr = N * r;
        realdp *A_U = new realdp[dr];
        realdp *A_V = new realdp[nr];
        for (integer i = 0; i < d * r; i++)
        {
            A_U[i] = genrandnormal();
        }
        for (integer i = 0; i < N * r; i++)
        {
            A_V[i] = genrandnormal();
        }
        
        realdp *V = new realdp[nz];
        for (integer i = 0; i < nz; i++)
        {
            V[i] = 0;
            for (integer j = 0; j < r; j++)
            {
                V[i] += A_U[ir[i] + j * d] * A_V[jc[i] + j * N];
            }
        }
        delete[]A_U;
        delete[]A_V;
        GrassMatCompletion Prob(ir, jc, V, nz, N, d, r);
        Prob.SetDomain(&Domain);
        realdp initstepsizes[8] = {0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.02};
        
        std::vector<std::string> names = {"RSGD", "RADAM", "RADAMSP", "RAMSGRAD", "RAMSGRADSP", "RSVRG", "SVRLRBFGS", "SVRLRBroydenFamily"};
        testStoSmoothProblem(&Prob, &GrassX, "testGrassMatCompletion", names, initstepsizes, 1);
        std::vector<std::string> names2 = {"RSD", "RCG", "LRBFGS", "RTRSD"};
        testSmoothProblem(&Prob, &GrassX, "testGrassMatCompletion", names2);
        
        delete[] V;
        delete[] ir;
    }
    
    {
        genrandseed(seed);
        printf("\n testSPDKarcherMean\n");

        /*Randomly generate a point on the SPD manifold*/
        integer n = 3, num = 30;

        // Define the manifold
        SPDManifold Domain(n);
        Domain.ChooseParamsSet3();
        Variable InitialX = Domain.RandominManifold();

        Vector EE(n, n), tmp(n, n);
        Vector Ls(1, &EE, num);
        Ls.NewMemoryOnWrite();
        for(integer i = 0; i < num; i++)
        {
            tmp.RandGaussian(); tmp.QRDecom();
            Ls.GetElement(i) = tmp.Field("_R").GetTranspose();
        }

        // Define the problem
        SPDKarcherMean Prob(Ls, n, num);
        /*The domain of the problem is a SPD manifold*/
        Prob.SetDomain(&Domain);
        
        realdp initstepsizes[8] = {0.002, 0.02, 0.004, 0.02, 0.02, 0.02, 0.02, 0.04};

        std::vector<std::string> names = {"RSGD", "RADAM", "RADAMSP", "RAMSGRAD", "RAMSGRADSP", "RSVRG", "SVRLRBFGS", "SVRLRBroydenFamily"};
        testStoSmoothProblem(&Prob, &InitialX, "testSPDKarcherMean", names, initstepsizes, 1);
        
        Domain.ChooseParamsSet1();
        std::vector<std::string> names2 = {"RSD", "RNewton", "RCG", "RBroydenFamily", "RBFGS", "LRBFGS", "LRBroydenFamily", "RTRSD", "RTRNewton", "RTRSR1", "LRTRSR1"};
        testSmoothProblem(&Prob, &InitialX, "testSPDKarcherMean", names2);
    }
    
    {
        genrandseed(seed);
        printf("\n testGrassPCA\n");
        
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
        
        realdp initstepsizes[8] = {0.002, 0.02, 0.02, 0.02, 0.02, 0.002, 0.002, 0.004};

        std::vector<std::string> names = {"RSGD", "RADAM", "RADAMSP", "RAMSGRAD", "RAMSGRADSP", "RSVRG", "SVRLRBFGS", "SVRLRBroydenFamily"};
        testStoSmoothProblem(&Prob, &GrassX, "testGrassPCA", names, initstepsizes, 1);
        std::vector<std::string> names2 = {"RSD", "RCG", "LRBFGS", "RTRSD"};
        testSmoothProblem(&Prob, &GrassX, "testGrassPCA", names2);
    }
    
    {
        genrandseed(seed);
        printf("\n testStieSoftICA\n");
        
        // size of the Stiefel manifold
        integer n = 6, p = 3;
        // number of covariance matrices
        integer N = 50;

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
        for (integer i = 0; i < Cs.Getlength(); i++)
        {
            Csptr[i] /= N;
        }
        
        Stiefel Domain(n, p);
        Domain.ChooseParamsSet3();
//        Domain.SetHasHHR(true);
        Vector InitialX = Domain.RandominManifold();
        StieSoftICA Prob(Cs, p);
        Prob.SetDomain(&Domain);

        realdp initstepsizes[8] = {0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.02};

        std::vector<std::string> names = {"RSGD", "RADAM", "RADAMSP", "RAMSGRAD", "RAMSGRADSP", "RSVRG", "SVRLRBFGS", "SVRLRBroydenFamily"};
        testStoSmoothProblem(&Prob, &InitialX, "testStieSoftICA", names, initstepsizes, 1);
        std::vector<std::string> names2 = {"RSD", "RCG", "LRBFGS", "RTRSD"};
        testSmoothProblem(&Prob, &InitialX, "testStieSoftICA", names2);
    }
    
    {
        genrandseed(seed);
        printf("\n testPoincareEmbeddings\n");
        
        /*Randomly generate a point on the Poincare Ball*/
        integer n = 5;
        integer numoftypes = 1, numofmani = 68;
        integer sampleNum = 50;
        // Define the manifold
        PoincareBall mani(n);
        mani.ChooseParamsSet1();
        ProductManifold Domain(numoftypes, &mani, numofmani);
        Variable ProdX = Domain.RandominManifold();
        
        // ------------------- read data -------------------
        std::string fname = "/Users/whuang/Documents/Syn/Codes/newROPTLIB/ROPTLIB/Matlab/ForCpp/wn_mini.csv"; //Relative path does not work properly. Absolute path is used.
        std::ifstream csv_data(fname, std::ios::in);
        std::string line;

        if (!csv_data.is_open())
        {
            std::cout << "Error: opening file fail" << std::endl;
        } else
        {
            std::istringstream sin;
            std::string word;
            Vector words(sampleNum * 2);
            realdp *wordsptr = words.ObtainWriteEntireData();
            std::getline(csv_data, line);
            integer i = 0;

            while (std::getline(csv_data, line))
            {
                sin.clear();
                sin.str(line);
                
                while (std::getline(sin, word, ','))
                {
                    wordsptr[i] = std::stol(word);
                    i++;
                }
            }
            csv_data.close();
            words.Reshape(2, sampleNum).Transpose();
            
            integer NegSampleNum = 4;
            realdp SampleDampening = 0.75;
            PoincareEmbeddings Prob(words, n, numofmani, NegSampleNum, SampleDampening);
            Prob.SetDomain(&Domain);
            realdp initstepsizes[8] = {3, 0.1, 0.1, 0.1, 0.1, 1, 0.008, 0.02};
            
            std::vector<std::string> names = {"RSGD", "RADAM", "RADAMSP", "RAMSGRAD", "RAMSGRADSP", "RSVRG", "SVRLRBFGS", "SVRLRBroydenFamily"};
            testStoSmoothProblem(&Prob, &ProdX, "testPoincareEmbeddings", names, initstepsizes, 1);
        }
    }
    
    {
        genrandseed(seed);
        printf("\n testFRankESparseApprox\n");
        integer m = 5, n = 5, r = 2;
        //integer m = 100, n = 15, r = 5;
        FixedRankE Domain(m, n, r);
        Domain.SetHasHHR(false);
        Variable InitialX = Domain.RandominManifold();

        Vector A(m, n); A.RandGaussian();
        realdp lambda = 0.1;
        integer lengthW = 1;

        FRankESparseApprox Prob(A, lambda, m, n, r, lengthW);
        Prob.SetDomain(&Domain);

        std::vector<std::string> names = {"IARPG", "IRPG"};
        testProxGradProblem(&Prob, &InitialX, "testFRankESparseApprox", names);
    }

    {
        genrandseed(seed);
        printf("\n testStieSPCA\n");
        // size of the Stiefel manifold
        integer n = 5, m = 4, p = 3;
        realdp lambda = 1;
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
        Stiefel Domain(n, p);
        Domain.ChooseParamsSet4();
        Variable StieX = Domain.RandominManifold();
        integer lengthW = 1;
        StieSPCA Prob(B, lambda, n, m, p, lengthW);
        /*The domain of the problem is a Stiefel manifold*/
        Prob.SetDomain(&Domain);
        std::vector<std::string> names = {"IARPG", "IRPG"};
        testProxGradProblem(&Prob, &StieX, "testStieSPCA", names);
    }

    {
        genrandseed(seed);
        printf("\n testSphereSparsestVector\n");
        // size of the matrix Q
        integer m = 10, n = 3;
        // Generate the matrix
        Vector Q(m, n);
        Q.RandGaussian();
        // Define the manifold
        Sphere Domain(n);
        Domain.SetHasHHR(true);
        Variable SphereX = Domain.RandominManifold();
        //Domain.SetHasHHR(true); /*set whether the manifold uses the idea in [HGA2015, Section 4.3] or not*/
        // Define the SparestVector problem
        SphereSparsestVector Prob(Q);
        /*The domain of the problem is a Stiefel manifold*/
        Prob.SetDomain(&Domain);
        std::vector<std::string> names = {"LRBFGSSub", "RBFGSSub", "RGS"};
        testSubGradProblem(&Prob, &SphereX, "testSphereSparsestVector", names);
    }
};

void testSubGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames)
{
    #ifdef SINGLE_PRECISION
        realdp tol = 5e-2;
    #else
        realdp tol = 1e-6;
    #endif
    if(stringinclude(Methodnames, "LRBFGSSub"))
    {
        printf("\n********************************Check LRBFGSSub in %s*************************************\n", probname);
        LRBFGSSub *LRBFGSSubsolver = new LRBFGSSub(prob, initx);
        LRBFGSSubsolver->Verbose = FINALRESULT;
        LRBFGSSubsolver->Tolerance = tol;
        LRBFGSSubsolver->Max_Iteration = 200;
    //    LRBFGSSubsolver->CheckParams();
        LRBFGSSubsolver->Run();
        if (LRBFGSSubsolver->Getnormndnd0() < tol)
            printf("SUCCESS!\n");
        else
            printf("FAIL!\n");
        delete LRBFGSSubsolver;
    }
    
    if(stringinclude(Methodnames, "RBFGSSub"))
    {
        printf("\n********************************Check RBFGSSub in %s*************************************\n", probname);
        RBFGSSub *RBFGSSubsolver = new RBFGSSub(prob, initx);
        RBFGSSubsolver->Verbose = FINALRESULT;
        RBFGSSubsolver->Tolerance = tol;
        RBFGSSubsolver->Max_Iteration = 200;
    //    RBFGSSubsolver->CheckParams();
        RBFGSSubsolver->Run();
        if (RBFGSSubsolver->Getnormndnd0() < tol)
            printf("SUCCESS!\n");
        else
            printf("FAIL!\n");
        delete RBFGSSubsolver;
    }
    
    if(stringinclude(Methodnames, "RGS"))
    {
        printf("\n********************************Check RGS in %s*************************************\n", probname);
        RGS *RGSsolver = new RGS(prob, initx);
        RGSsolver->Verbose = FINALRESULT;
        RGSsolver->Tolerance = tol;
        RGSsolver->Max_Iteration = 200;
    //    RGSsolver->CheckParams();
        RGSsolver->Run();
        if (RGSsolver->Getnormndnd0() < tol)
            printf("SUCCESS!\n");
        else
            printf("FAIL!\n");
        delete RGSsolver;
    }
};

void testProxGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames)
{
    #ifdef SINGLE_PRECISION
        realdp tol = 1e-1;
    #else
        realdp tol = 1e-4;
    #endif
    if(stringinclude(Methodnames, "IARPG"))
    {
        printf("\n********************************Check IARPG in %s*************************************\n", probname);
        IARPG *IARPGsolver = new IARPG(prob, initx);
        IARPGsolver->Max_Iteration = 500;
        IARPGsolver->Verbose = FINALRESULT;
        IARPGsolver->Tolerance = tol;
    //    IARPGsolver->Variant = LSPG_REGULAR; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    //    IARPGsolver->CheckParams();
        IARPGsolver->Run();
        
        if (IARPGsolver->Getnormndnd0() < tol)
            printf("SUCCESS!\n");
        else
            printf("FAIL!\n");

        delete IARPGsolver;
    }
    
    if(stringinclude(Methodnames, "IRPG"))
    {
        printf("\n********************************Check IRPG in %s*************************************\n", probname);
        IRPG *IRPGsolver = new IRPG(prob, initx);
        IRPGsolver->Max_Iteration = 500;
        IRPGsolver->Verbose = FINALRESULT;
        IRPGsolver->Tolerance = tol;
    //    IRPGsolver->Variant = LSPG_ADALIPSCHITZ; //-- LSPG_REGULAR; //-- LSPG_ADALIPSCHITZ;
    //    IRPGsolver->CheckParams();
        IRPGsolver->Run();
        
        if (IRPGsolver->Getnormndnd0() < tol)
            printf("SUCCESS!\n");
        else
            printf("FAIL!\n");

        delete IRPGsolver;
    }
};

void testStoSmoothProblem(Problem *Prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, realdp initstepsizes[], integer numGradHess)
{
//#ifdef SINGLE_PRECISION
//    realdp tol = 5e-2;
//#else
//    realdp tol = 1e-6;
//#endif
    
    realdp tol = 1e-2; // Stochastic gradient algorithms converge slowly. Therefore, the stopping criterion is not strict.
    integer idx = 0;
    // test RSGD
    if(stringinclude(Methodnames, "RSGD"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RSGD in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RSGD *RSGDsolver = new RSGD(Prob, initx);
            RSGDsolver->Verbose = FINALRESULT;
            RSGDsolver->Max_Iteration = 3000;
            RSGDsolver->Tolerance = tol;
            RSGDsolver->Initstepsize = initstepsizes[idx];
            RSGDsolver->Run();
            if (RSGDsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RSGDsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RSGD in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RSGD *RSGDsolver = new RSGD(Prob, initx);
            RSGDsolver->Verbose = FINALRESULT;
            RSGDsolver->Max_Iteration = 3000;
            RSGDsolver->Tolerance = tol;
            RSGDsolver->Initstepsize = initstepsizes[idx];
            RSGDsolver->Run();
            if (RSGDsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RSGDsolver;
        }
        idx++;
    }
    
    // test RADAM
    if(stringinclude(Methodnames, "RADAM"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RADAM in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RADAM *RADAMsolver = new RADAM(Prob, initx);
            RADAMsolver->Verbose = FINALRESULT;
            RADAMsolver->Max_Iteration = 3000;
            RADAMsolver->Tolerance = tol;
            RADAMsolver->Initstepsize = initstepsizes[idx];
            RADAMsolver->Run();
            if (RADAMsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RADAMsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RADAM in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RADAM *RADAMsolver = new RADAM(Prob, initx);
            RADAMsolver->Verbose = FINALRESULT;
            RADAMsolver->Max_Iteration = 3000;
            RADAMsolver->Tolerance = tol;
            RADAMsolver->Initstepsize = initstepsizes[idx];
            RADAMsolver->Run();
            if (RADAMsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RADAMsolver;
        }
        idx++;
    }
    
    // test RADAMSP
    if(stringinclude(Methodnames, "RADAMSP"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RADAMSP in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RADAMSP *RADAMSPsolver = new RADAMSP(Prob, initx);
            RADAMSPsolver->Verbose = FINALRESULT;
            RADAMSPsolver->Max_Iteration = 3000;
            RADAMSPsolver->Initstepsize = initstepsizes[idx]; //--0.1;
//            RADAMSPsolver->burnin_multiplier = 3;//--0.08;
//            RADAMSPsolver->NumFixedStep = 20;
//            RADAMSPsolver->isFixed = false;
            RADAMSPsolver->Tolerance = tol;
            RADAMSPsolver->Run();
            if (RADAMSPsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RADAMSPsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RADAMSP in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RADAMSP *RADAMSPsolver = new RADAMSP(Prob, initx);
            RADAMSPsolver->Verbose = FINALRESULT;
            RADAMSPsolver->Max_Iteration = 3000;
            RADAMSPsolver->Initstepsize = initstepsizes[idx]; //--0.1;
//            RADAMSPsolver->burnin_multiplier = 3;//--0.08;
//            RADAMSPsolver->NumFixedStep = 20;
//            RADAMSPsolver->isFixed = false;
            RADAMSPsolver->Tolerance = tol;
            RADAMSPsolver->Run();
            if (RADAMSPsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RADAMSPsolver;
        }
        idx++;
    }
    
    // test RAMSGRAD
    if(stringinclude(Methodnames, "RAMSGRAD"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RAMSGRAD in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RAMSGRAD *RAMSGRADsolver = new RAMSGRAD(Prob, initx);
            RAMSGRADsolver->Verbose = FINALRESULT;
            RAMSGRADsolver->Max_Iteration = 3000;
            RAMSGRADsolver->Tolerance = tol;
            RAMSGRADsolver->Initstepsize = initstepsizes[idx];
            RAMSGRADsolver->Run();
            if (RAMSGRADsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RAMSGRADsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RAMSGRAD in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RAMSGRAD *RAMSGRADsolver = new RAMSGRAD(Prob, initx);
            RAMSGRADsolver->Verbose = FINALRESULT;
            RAMSGRADsolver->Max_Iteration = 3000;
            RAMSGRADsolver->Tolerance = tol;
            RAMSGRADsolver->Initstepsize = initstepsizes[idx];
            RAMSGRADsolver->Run();
            if (RAMSGRADsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RAMSGRADsolver;
        }
        idx++;
    }
    
    // test RAMSGRAD
    if(stringinclude(Methodnames, "RAMSGRADSP"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RAMSGRADSP in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RAMSGRADSP *RAMSGRADSPsolver = new RAMSGRADSP(Prob, initx);
            RAMSGRADSPsolver->Verbose = FINALRESULT;
            RAMSGRADSPsolver->Max_Iteration = 3000;
            RAMSGRADSPsolver->Initstepsize = initstepsizes[idx]; //-- 0.1;
//            RAMSGRADSPsolver->gamma = 0.1;
//            RAMSGRADSPsolver->C = 10;
//            RAMSGRADSPsolver->theta = 0;
//            RAMSGRADSPsolver->Tolerance = static_cast<realdp> (1e-6);
//            RAMSGRADSPsolver->isFixed = true;
//            RAMSGRADSPsolver->NumFixedStep = 20;
//            RAMSGRADSPsolver->burnin_multiplier = 0.01;
            RAMSGRADSPsolver->Tolerance = tol;
            RAMSGRADSPsolver->Run();
            if (RAMSGRADSPsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RAMSGRADSPsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RAMSGRADSP in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RAMSGRADSP *RAMSGRADSPsolver = new RAMSGRADSP(Prob, initx);
            RAMSGRADSPsolver->Verbose = FINALRESULT;
            RAMSGRADSPsolver->Max_Iteration = 3000;
            RAMSGRADSPsolver->Initstepsize = initstepsizes[idx]; //-- 0.1;
//            RAMSGRADSPsolver->gamma = 0.1;
//            RAMSGRADSPsolver->C = 10;
//            RAMSGRADSPsolver->theta = 0;
//            RAMSGRADSPsolver->Tolerance = static_cast<realdp> (1e-6);
//            RAMSGRADSPsolver->isFixed = true;
//            RAMSGRADSPsolver->NumFixedStep = 20;
//            RAMSGRADSPsolver->burnin_multiplier = 0.01;
            RAMSGRADSPsolver->Tolerance = tol;
            RAMSGRADSPsolver->Run();
            if (RAMSGRADSPsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RAMSGRADSPsolver;
        }
        idx++;
    }
    
    // test RSVRG
    if(stringinclude(Methodnames, "RSVRG"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check RSVRG in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            RSVRG *RSVRGsolver = new RSVRG(Prob, initx);
            RSVRGsolver->Verbose = FINALRESULT;
            RSVRGsolver->Max_Iteration = 3000;
            RSVRGsolver->Initstepsize = initstepsizes[idx]; //-- 1;
            RSVRGsolver->Tolerance = tol;
            RSVRGsolver->Run();
            if (RSVRGsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RSVRGsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check RSVRG in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            RSVRG *RSVRGsolver = new RSVRG(Prob, initx);
            RSVRGsolver->Verbose = FINALRESULT;
            RSVRGsolver->Max_Iteration = 3000;
            RSVRGsolver->Initstepsize = initstepsizes[idx]; //-- 1;
            RSVRGsolver->Tolerance = tol;
            RSVRGsolver->Run();
            if (RSVRGsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete RSVRGsolver;
        }
        idx++;
    }
    
    // test SVRLRBFGS
    if(stringinclude(Methodnames, "SVRLRBFGS"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check SVRLRBFGS in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            SVRLRBFGS *SVRLRBFGSsolver = new SVRLRBFGS(Prob, initx);
            SVRLRBFGSsolver->Verbose = FINALRESULT;
            SVRLRBFGSsolver->Max_Iteration = 3000;
            SVRLRBFGSsolver->Initstepsize = initstepsizes[idx]; //-- 1e-2;
            SVRLRBFGSsolver->Tolerance = tol;
            SVRLRBFGSsolver->Run();
            if (SVRLRBFGSsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete SVRLRBFGSsolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check SVRLRBFGS in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            SVRLRBFGS *SVRLRBFGSsolver = new SVRLRBFGS(Prob, initx);
            SVRLRBFGSsolver->Verbose = FINALRESULT;
            SVRLRBFGSsolver->Max_Iteration = 3000;
            SVRLRBFGSsolver->Initstepsize = initstepsizes[idx]; //-- 1e-2;
            SVRLRBFGSsolver->Tolerance = tol;
            SVRLRBFGSsolver->Run();
            if (SVRLRBFGSsolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete SVRLRBFGSsolver;
        }
        idx++;
    }
    
    // test SVRLRBroydenFamily
    if(stringinclude(Methodnames, "SVRLRBroydenFamily"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check SVRLRBroydenFamilysolver in %s*****************************************\n", probname);
            Prob->SetNumGradHess(false);
            SVRLRBroydenFamily *SVRLRBroydenFamilysolver = new SVRLRBroydenFamily(Prob, initx);
            SVRLRBroydenFamilysolver->Verbose = FINALRESULT;
            SVRLRBroydenFamilysolver->Max_Iteration = 3000;
            SVRLRBroydenFamilysolver->Initstepsize = initstepsizes[idx];//-- 2e-2;
            SVRLRBroydenFamilysolver->Tolerance = tol;
            SVRLRBroydenFamilysolver->Run();
            if (SVRLRBroydenFamilysolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete SVRLRBroydenFamilysolver;
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check SVRLRBroydenFamilysolver in %s with numerical gradient and Hessian*****************************************\n", probname);
            Prob->SetNumGradHess(true);
            SVRLRBroydenFamily *SVRLRBroydenFamilysolver = new SVRLRBroydenFamily(Prob, initx);
            SVRLRBroydenFamilysolver->Verbose = FINALRESULT;
            SVRLRBroydenFamilysolver->Max_Iteration = 3000;
            SVRLRBroydenFamilysolver->Initstepsize = initstepsizes[idx];//-- 2e-2;
            SVRLRBroydenFamilysolver->Tolerance = tol;
            SVRLRBroydenFamilysolver->Run();
            if (SVRLRBroydenFamilysolver->Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            delete SVRLRBroydenFamilysolver;
        }
    }
};

void testSmoothProblem(Problem *Prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, integer numGradHess)
{
#ifdef SINGLE_PRECISION
    realdp tol = 5e-2;
#else
    realdp tol = 1e-5;
#endif
    // test RSD
    if(stringinclude(Methodnames, "RSD"))
    {
        integer RSDMI[5] = { 2000, 2000, 2000, 2000, 2000 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("********************************Check all line search algorithm in RSD in %s*****************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++) //LSSM_INPUTFUN
            {
                Prob->SetNumGradHess(false);
                RSD *RSDsolver = new RSD(Prob, initx);
                RSDsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RSDsolver->Verbose = FINALRESULT;
                RSDsolver->Max_Iteration = RSDMI[i];
                RSDsolver->Run();
                if (RSDsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RSDsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("********************************Check all line search algorithm in RSD in %s with numerical gradient and Hessian*****************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(true);
                RSD *RSDsolver = new RSD(Prob, initx);
                RSDsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RSDsolver->Verbose = FINALRESULT;
                RSDsolver->Max_Iteration = RSDMI[i];
                RSDsolver->Run();
                if (RSDsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RSDsolver;
            }
        }
    }
    
    // test RNewton
    if(stringinclude(Methodnames, "RNewton"))
    {
        integer RNewtonMI[5] = { 80, 80, 80, 80, 80 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all line search algorithm in RNewton in %s*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++) // LSALGOLENGTH
            {
                Prob->SetNumGradHess(false);
                RNewton *RNewtonsolver = new RNewton(Prob, initx);
                RNewtonsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RNewtonsolver->Verbose = FINALRESULT;
                /*Uncomment following two lines to use the linesearch algorithm defined by the function "LinesearchInput".*/
                //RNewtonsolver->LineSearch_LS = INPUTFUN;
                //RNewtonsolver->LinesearchInput = &LinesearchInput;
                RNewtonsolver->Max_Iteration = RNewtonMI[i];
//                RNewtonsolver->Verbose = ITERRESULT;//---
//                RNewtonsolver->CheckParams();//---
                RNewtonsolver->Run();

                if (RNewtonsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete RNewtonsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all line search algorithm in RNewton in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++) //LSSM_INPUTFUN  LSALGOLENGTH
            {
                Prob->SetNumGradHess(true);
                RNewton *RNewtonsolver = new RNewton(Prob, initx);
                RNewtonsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RNewtonsolver->Verbose = FINALRESULT;
                /*Uncomment following two lines to use the linesearch algorithm defined by the function "LinesearchInput".*/
                //RNewtonsolver->LineSearch_LS = INPUTFUN;
                //RNewtonsolver->LinesearchInput = &LinesearchInput;
                RNewtonsolver->Max_Iteration = RNewtonMI[i];
                //RNewtonsolver->CheckParams();
                RNewtonsolver->Run();

                if (RNewtonsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete RNewtonsolver;
            }
        }
    }

    // test RCG
    if(stringinclude(Methodnames, "RCG"))
    {
        integer RCGMI[6] = { 2000, 2000, 2000, 2000, 2000, 2000 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all Formulas in RCG in %s*************************************\n", probname);
            for (integer i = 0; i < RCGMETHODSLENGTH; i++)
            {
                Prob->SetNumGradHess(false);
//                Prob->CheckGradHessian(*initx);//--
//                Prob->GetDomain()->CheckParams();//--
                RCG *RCGsolver = new RCG(Prob, initx);
                RCGsolver->RCGmethod = static_cast<RCGmethods> (i);
                //RCGsolver->LineSearch_LS = LSSM_ARMIJO;
                //RCGsolver->InitSteptype = QUADINTMOD;
                RCGsolver->Verbose = FINALRESULT;
                RCGsolver->Max_Iteration = RCGMI[i];
//                RCGsolver->CheckParams();
                RCGsolver->Run();
                if (RCGsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RCGsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all Formulas in RCG in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < RCGMETHODSLENGTH; i++)
            {
                Prob->SetNumGradHess(true);
                RCG *RCGsolver = new RCG(Prob, initx);
                RCGsolver->RCGmethod = static_cast<RCGmethods> (i);
                //RCGsolver->LineSearch_LS = LSSM_ARMIJO;
                //RCGsolver->InitSteptype = QUADINTMOD;
                RCGsolver->Verbose = FINALRESULT;
                RCGsolver->Max_Iteration = RCGMI[i];
                //RCGsolver->CheckParams();
                RCGsolver->Run();
                if (RCGsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RCGsolver;
            }
        }
    }
    
    // test RBroydenFamily
    if(stringinclude(Methodnames, "RBroydenFamily"))
    {
        integer RBroydenFamilyMI[5] = { 200, 200, 200, 200, 200 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all Formulas in RBroydenFamily in %s*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(false);
                RBroydenFamily *RBroydenFamilysolver = new RBroydenFamily(Prob, initx);
                RBroydenFamilysolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RBroydenFamilysolver->Verbose = FINALRESULT;
                RBroydenFamilysolver->Max_Iteration = RBroydenFamilyMI[i];
                RBroydenFamilysolver->Run();
                if (RBroydenFamilysolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RBroydenFamilysolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all Formulas in RBroydenFamily in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(true);
                RBroydenFamily *RBroydenFamilysolver = new RBroydenFamily(Prob, initx);
                RBroydenFamilysolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RBroydenFamilysolver->Verbose = FINALRESULT;
                RBroydenFamilysolver->Max_Iteration = RBroydenFamilyMI[i];
                //RBroydenFamilysolver->CheckParams();
                RBroydenFamilysolver->Run();
                if (RBroydenFamilysolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RBroydenFamilysolver;
            }
        }
    }
    
    // test RWRBFGS
    if(stringinclude(Methodnames, "RWRBFGS"))
    {
        integer RWRBFGSMI[5] = { 200, 200, 200, 200, 200 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all line search algorithm in RWRBFGS in %s*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(false);
                RWRBFGS *RWRBFGSsolver = new RWRBFGS(Prob, initx);
                RWRBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RWRBFGSsolver->Verbose = FINALRESULT; //ITERRESULT;//
                RWRBFGSsolver->Max_Iteration = RWRBFGSMI[i];
                //RWRBFGSsolver->CheckParams();
                RWRBFGSsolver->Run();
                if (RWRBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RWRBFGSsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all line search algorithm in RWRBFGS in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(true);
                RWRBFGS *RWRBFGSsolver = new RWRBFGS(Prob, initx);
                RWRBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RWRBFGSsolver->Verbose = FINALRESULT; //ITERRESULT;//
                RWRBFGSsolver->Max_Iteration = RWRBFGSMI[i];
                //RWRBFGSsolver->CheckParams();
                RWRBFGSsolver->Run();
                if (RWRBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RWRBFGSsolver;
            }
        }
    }
    
    // test RBFGS
    if(stringinclude(Methodnames, "RBFGS"))
    {
        integer RBFGSMI[5] = { 200, 200, 200, 200, 200 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all line search algorithm in RBFGS in %s*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(false);
                RBFGS *RBFGSsolver = new RBFGS(Prob, initx);
                RBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RBFGSsolver->Verbose = FINALRESULT;
                RBFGSsolver->Max_Iteration = RBFGSMI[i];
                //RBFGSsolver->CheckParams();
                RBFGSsolver->Run();
                if (RBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RBFGSsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all line search algorithm in RBFGS in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)
            {
                Prob->SetNumGradHess(true);
                RBFGS *RBFGSsolver = new RBFGS(Prob, initx);
                RBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                RBFGSsolver->Verbose = FINALRESULT;
                RBFGSsolver->Max_Iteration = RBFGSMI[i];
                //RBFGSsolver->CheckParams();
                RBFGSsolver->Run();
                if (RBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");
                delete RBFGSsolver;
            }
        }
    }
    
    // test LRBFGS
    if(stringinclude(Methodnames, "LRBFGS"))
    {
        integer LRBFGSMI[5] = { 200, 200, 200, 200, 200 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all line search algorithm in LRBFGS in %s*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)//
            {
                Prob->SetNumGradHess(false);
                LRBFGS *LRBFGSsolver = new LRBFGS(Prob, initx);
                LRBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                LRBFGSsolver->Verbose = FINALRESULT; //ITERRESULT;//
                LRBFGSsolver->Max_Iteration = LRBFGSMI[i];
                //LRBFGSsolver->CheckParams();
                LRBFGSsolver->Run();

                if (LRBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete LRBFGSsolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all line search algorithm in LRBFGS in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)//
            {
                Prob->SetNumGradHess(true);
                LRBFGS *LRBFGSsolver = new LRBFGS(Prob, initx);
                LRBFGSsolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                LRBFGSsolver->Verbose = FINALRESULT; //ITERRESULT;//
                LRBFGSsolver->Max_Iteration = LRBFGSMI[i];
//                LRBFGSsolver->Verbose = ITERRESULT;//---
                //LRBFGSsolver->CheckParams();
                LRBFGSsolver->Run();

                if (LRBFGSsolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete LRBFGSsolver;
            }
        }
    }
    
    // test LRBroydenFamily
    if(stringinclude(Methodnames, "LRBroydenFamily"))
    {
        integer LRBroydenFamilyMI[5] = { 200, 200, 200, 200, 200 };
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check all line search algorithm in LRBFGS in %s*************************************\n", probname);
//            for (integer i = 1; i < 2; i++)//
            for (integer i = 0; i < LSSM_INPUTFUN; i++)//
            {
                Prob->SetNumGradHess(false);
                LRBroydenFamily *LRBroydenFamilysolver = new LRBroydenFamily(Prob, initx);
                LRBroydenFamilysolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                LRBroydenFamilysolver->Verbose = FINALRESULT; //ITERRESULT;//
                LRBroydenFamilysolver->Max_Iteration = LRBroydenFamilyMI[i];
//                LRBroydenFamilysolver->Verbose = ITERRESULT;//---
//                LRBroydenFamilysolver->CheckParams();//---
                LRBroydenFamilysolver->Run();

                if (LRBroydenFamilysolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete LRBroydenFamilysolver;
            }
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check all line search algorithm in LRBFGS in %s with numerical gradient and Hessian*************************************\n", probname);
            for (integer i = 0; i < LSSM_INPUTFUN; i++)//
            {
                Prob->SetNumGradHess(true);
                LRBroydenFamily *LRBroydenFamilysolver = new LRBroydenFamily(Prob, initx);
                LRBroydenFamilysolver->LineSearch_LS = static_cast<LSAlgoSM> (i);
                LRBroydenFamilysolver->Verbose = FINALRESULT; //ITERRESULT;//
                LRBroydenFamilysolver->Max_Iteration = LRBroydenFamilyMI[i];
                //LRBroydenFamilysolver->CheckParams();
                LRBroydenFamilysolver->Run();

                if (LRBroydenFamilysolver->Getnormgfgf0() < tol)
                    printf("SUCCESS!\n");
                else
                    printf("FAIL!\n");

                delete LRBroydenFamilysolver;
            }
        }
    }
    // test RTRSD
    if(stringinclude(Methodnames, "RTRSD"))
    {
        
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check RTRSD in %s*************************************\n", probname);
            Prob->SetNumGradHess(false);
            RTRSD RTRSDsolver(Prob, initx);
            RTRSDsolver.Verbose = FINALRESULT;
            RTRSDsolver.kappa = 0.001;
            RTRSDsolver.Max_Iteration = 5000;
            //RTRSDsolver.CheckParams();
            RTRSDsolver.Run();
            if (RTRSDsolver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
        
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check RTRSD in %s with numerical gradient and Hessian*************************************\n", probname);
            Prob->SetNumGradHess(true);
            RTRSD RTRSDsolver(Prob, initx);
            RTRSDsolver.Verbose = FINALRESULT;
            RTRSDsolver.kappa = 0.001;
            RTRSDsolver.Max_Iteration = 5000;
            //RTRSDsolver.CheckParams();
            RTRSDsolver.Run();
            if (RTRSDsolver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
    }
    
    // test RTRNewton
    if(stringinclude(Methodnames, "RTRNewton"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check RTRNewton in %s*************************************\n", probname);
            Prob->SetNumGradHess(false);
            RTRNewton RTRNewtonsolver(Prob, initx);
            RTRNewtonsolver.Verbose = FINALRESULT;
            RTRNewtonsolver.Max_Iteration = 50;
            //RTRNewtonsolver.CheckParams();
            RTRNewtonsolver.Run();
            if (RTRNewtonsolver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
            Prob->MinMaxEigValHess(RTRNewtonsolver.GetXopt()).Print("Min and Max of the eigenvalues of Hessian at x_k");
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check RTRNewton in %s with numerical gradient and Hessian*************************************\n", probname);
            Prob->SetNumGradHess(true);
            RTRNewton RTRNewtonsolver(Prob, initx);
            RTRNewtonsolver.Verbose = FINALRESULT;
            RTRNewtonsolver.Max_Iteration = 50;
            //RTRNewtonsolver.CheckParams();
            RTRNewtonsolver.Run();
            if (RTRNewtonsolver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
    }
    
    // test RTRSR1
    if(stringinclude(Methodnames, "RTRSR1"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check RTRSR1 in %s*************************************\n", probname);
            Prob->SetNumGradHess(false);
            RTRSR1 RTRSR1solver(Prob, initx);
            RTRSR1solver.Verbose = FINALRESULT;
            RTRSR1solver.Max_Iteration = 200;
            //RTRSR1solver.CheckParams();
            RTRSR1solver.Run();
            if (RTRSR1solver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check RTRSR1 in %s with numerical gradient and Hessian*************************************\n", probname);
            Prob->SetNumGradHess(true);
            RTRSR1 RTRSR1solver(Prob, initx);
            RTRSR1solver.Verbose = FINALRESULT;
            RTRSR1solver.Max_Iteration = 200;
            //RTRSR1solver.CheckParams();
            RTRSR1solver.Run();
            if (RTRSR1solver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
    }
    
    // test LRTRSR1
    if(stringinclude(Methodnames, "LRTRSR1"))
    {
        if(numGradHess == 0 || numGradHess == 1)
        {
            printf("\n********************************Check LRTRSR1 in %s*************************************\n", probname);
            Prob->SetNumGradHess(false);
            LRTRSR1 LRTRSR1solver(Prob, initx);
            LRTRSR1solver.OutputGap = 1;
            LRTRSR1solver.Max_Iteration = 500;
            //LRTRSR1solver.Shrinked_tau = 0.1;
            //LRTRSR1solver.LengthSY = 4;
            LRTRSR1solver.Verbose = FINALRESULT;//-- FINALRESULT;
            //LRTRSR1solver.CheckParams();
            LRTRSR1solver.Run();
            if (LRTRSR1solver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
        if(numGradHess == 0 || numGradHess == 2)
        {
            printf("\n********************************Check LRTRSR1 in %s with numerical gradient and Hessian*************************************\n", probname);
            Prob->SetNumGradHess(true);
            LRTRSR1 LRTRSR1solver(Prob, initx);
            LRTRSR1solver.OutputGap = 1;
            LRTRSR1solver.Max_Iteration = 500;
            //LRTRSR1solver.Shrinked_tau = 0.1;
            //LRTRSR1solver.LengthSY = 4;
            LRTRSR1solver.Verbose = FINALRESULT;//-- FINALRESULT;
            //LRTRSR1solver.CheckParams();
            LRTRSR1solver.Run();
            if (LRTRSR1solver.Getnormgfgf0() < tol)
                printf("SUCCESS!\n");
            else
                printf("FAIL!\n");
        }
    }
}

bool stringinclude(std::vector<std::string> names, std::string name)
{
    for(integer i = 0; i < names.size(); i++)
        if(strcmp(names[i].c_str(), name.c_str()) == 0)
            return true;
    return false;
}
