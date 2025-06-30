
#include "Solvers/LRTRSR1.h"

/*Define the namespace*/
namespace ROPTLIB{

    LRTRSR1::LRTRSR1(const Problem *prob, const Variable *initialx)
    {
        Initialization(prob, initialx);
    };

    void LRTRSR1::SetProbX(const Problem *prob, const Variable *initialx)
    {
        SolversSMTR::SetProbX(prob, initialx);
        prob->SetUseGrad(true);
        prob->SetUseHess(false);
        s = Prob->GetDomain()->GetEMPTY();
        y = Prob->GetDomain()->GetEMPTY();
        PMGQ = Vector();
        PsiTPsi = Vector();
    };

    void LRTRSR1::SetDefaultParams(void)
    {
        SolversSMTR::SetDefaultParams();
        theta = static_cast<realdp> (0.1);
        kappa = static_cast<realdp> (0.1);
        LengthSY = 4;
//        S = nullptr;
//        Y = nullptr;
        YMGS = nullptr;
        inpss = 0;
        inpsy = 0;
        inpyy = 0;
        Currentlength = 0;
        beginidx = 0;
//        SS = nullptr;
//        SY = nullptr;
        gamma = 1;
        Lgamma = 10000;
        SolverName.assign("LRTRSR1");
        innerIter = 0;
    };

    void LRTRSR1::SetParams(PARAMSMAP params)
    {
        SolversSMTR::SetParams(params);
        PARAMSMAP::iterator iter;
        for (iter = params.begin(); iter != params.end(); iter++)
        {
            if (iter->first == static_cast<std::string> ("LengthSY"))
            {
                LengthSY = static_cast<integer> (iter->second);
            }
            else
            if (iter->first == static_cast<std::string> ("Lgamma"))
            {
                Lgamma = static_cast<realdp> (iter->second);
            }
        }
    };

    LRTRSR1::~LRTRSR1(void)
    {
//        DeleteVectors(S, LengthSY);
//        DeleteVectors(Y, LengthSY);
        DeleteVectors(YMGS, LengthSY);
//        if (SS != nullptr)
//            delete[] SS;
//        SS = nullptr;
//        if (SY != nullptr)
//            delete[] SY;
//        SY = nullptr;
    };

    void LRTRSR1::Run(void)
    {
        if(! Mani->GetIsIntrinsic())
        {
            printf("Error: Use intrinsic representation! LRTRSR1 only supports intrinsic representation for efficiently solving the subproblem!\n");
            return;
        }
        
        DeleteVectors(YMGS, LengthSY);
        NewVectors(YMGS, LengthSY);
        
        
//        DeleteVectors(S, LengthSY);
//        NewVectors(S, LengthSY);
//        DeleteVectors(Y, LengthSY);
//        NewVectors(Y, LengthSY);
//        DeleteVectors(YMGS, LengthSY);
//        NewVectors(YMGS, LengthSY);
        
//        if (SS != nullptr)
//            delete[] SS;
//        SS = new realdp[LengthSY * LengthSY];
//        if (SY != nullptr)
//            delete[] SY;
//        SY = new realdp[LengthSY * LengthSY];
        SolversSMTR::Run();
    };

    void LRTRSR1::tCG_TR(void)
    {/*This method is only used when the Euclidean metric or an intrinsic approach is used.*/
        /*note that if the manifold is a complex, then we still view it as a real manifold with real reparameterizations
        therefore, we first set the types of vectors to be real and then recover the types of them at the end of this function*/
        bool gf1iscomplex = gf1.Getiscomplex(); gf1.Setiscomplex(false);
        
        integer n = gf1.Getlength(); /*number of rows*/
        integer k = Currentlength;    /*number of columns*/
        integer idx;
#ifdef SINGLE_PRECISION
        realdp tolLocalNewton = 1e-5;
#else
        realdp tolLocalNewton = 1e-10;
#endif
        Vector Psi(n, k);
        Vector R(k, k);
        realdp *Psiptr = Psi.ObtainWriteEntireData();
        
        /*Compute Psi = Y - gamma S*/
        for (integer i = 0; i < k; i++)
        {
            idx = (i + beginidx) % LengthSY;
            copy_(&n, const_cast<realdp *> (YMGS[idx].ObtainReadData()), &GLOBAL::IONE, Psiptr + i * n, &GLOBAL::IONE);
        }
        
        /* Cholesky decomposition for Psi^T Psi. If it fails due to numerical errors, then use QR decomposion Psikk = Q RT^T
        eigenvalue decomposition: U D U^T = eig(RT^T * PMGQ^{-1} RT)
        P_parallel = Q * U
        Therefore: Psi * (PMGQ)^{-1} * Psi = Q * U * D * U^T * Q^T = P_parallel * D * P_parallel^T */
        Vector RinvPMGQRT, P_parallel(Psi), D, Psig(Psi.Getcol(), 1), g_parallel(Psi.Getcol(), 1);
        if(k > 0)
        {
            if(PsiTPsi.CholDecom() == 0)
            { /* efficient but less numerical stability */
                R = PsiTPsi.Field("_L").GetTranspose();
            }
            else
            { /* expensive but numerical stability */
                Psi.QRDecom();
                R = Psi.Field("_R");
//                std::cout << "QR" << std::endl;//---
            }
            
            RinvPMGQRT = R * (PMGQ % R.GetTranspose());
            RinvPMGQRT.EigenDecomSym();

            P_parallel.AlphaABaddBetaThis(1, Psi, GLOBAL::N, R % RinvPMGQRT.Field("_EigVec"), GLOBAL::N, 0); /* P_parallel = Psi * R^{-1} * RinvPMGQRT.Field("_EigVec") = Psi.Field("_Q") * RinvPMGQRT.Field("_EigVec"); */
//            P_parallel.Print("P_parallel -1:");//---
            D = RinvPMGQRT.Field("_EigVal");
            Vector gf1reshape(gf1); gf1reshape.Reshape(n);
            Psig.AlphaABaddBetaThis(1, Psi, GLOBAL::T, gf1reshape, GLOBAL::N, 0); /* Psig = Psi.GetTranspose() * gf1.GetReshape(n); */
            g_parallel.AlphaABaddBetaThis(1, P_parallel, GLOBAL::T, gf1reshape, GLOBAL::N, 0); /* g_parallel = P_parallel.GetTranspose() * gf1.GetReshape(n); */
        }

//        P_parallel.Print("P_parallel 0:");//---
        
        Vector Lambda(k + 1);
        realdp *Lambdaptr = Lambda.ObtainWriteEntireData();
        const realdp *Dptr = D.ObtainReadData();
        for (integer i = 0; i < k; i++)
            Lambdaptr[i] = Dptr[i] + gamma;
        Lambdaptr[k] = gamma;
        
        for (integer i = 0; i < k + 1; i++)
            if (fabs(Lambdaptr[i]) <= tolLocalNewton)
                Lambdaptr[i] = 0;
        
        realdp lambda_min = Lambdaptr[0];
        for(integer i = 0; i < k + 1; i++)
            if(lambda_min > Lambdaptr[i])
                lambda_min = Lambdaptr[i];
        realdp a_kp2 = std::sqrt(std::fabs(Mani->Metric(x1, gf1, gf1) - ((k == 0) ? 0 : g_parallel.DotProduct(g_parallel))));
        
//        std::cout << Mani->Metric(x1, gf1, gf1) << ":" << ((k == 0) ? 0 : g_parallel.DotProduct(g_parallel)) << ":" << Mani->Metric(x1, gf1, gf1) - ((k == 0) ? 0 : g_parallel.DotProduct(g_parallel)) << std::endl;//---
//        std::cout << "a_kp2:" << a_kp2 << std::endl;//----
        
        if (a_kp2 * a_kp2 < tolLocalNewton)
            a_kp2 = 0;
        
        Vector a_j(k + 1);
        realdp *a_jptr = a_j.ObtainWriteEntireData();
        const realdp *g_parallelptr = g_parallel.ObtainReadData();
        
        for (integer i = 0; i < k; i++)
            a_jptr[i] = g_parallelptr[i];
        a_jptr[k] = a_kp2;
        realdp tmpv = 0;
        for (integer i = 0; i < k + 1; i++)
            tmpv += a_jptr[i] * a_jptr[i] / Lambdaptr[i] / Lambdaptr[i];
        tmpv = sqrt(tmpv);

        realdp *pStar = eta2.ObtainWriteEntireData();

        realdp sigmaStar = 0;
        tCGstatusSM = TRSM_MIN;
        
//        Lambda.Print("Lambda:");//---
//        a_j.Print("a_j:");//---
//        std::cout << "lambda_min:" << lambda_min << ", tmpv:" << tmpv << ", Delta:" << Delta << std::endl;//---
        if (lambda_min > 0 && tmpv <= Delta)
        {
            if(k == 0)
                ComputeSBySMW(gamma, Psig, Vector (), PMGQ, Psi);
            else
                ComputeSBySMW(gamma, Psig, R, PMGQ, Psi); /*gf1, Psig, YMGS, PMGQ, RT*/
            
//            std::cout << "lambda_min > 0 && tmpv <= Delta" << std::endl;//---
//            std::cout << "t1" << std::endl;//----
        }
        else
            if (lambda_min <= 0 && PhiBar_fg(-lambda_min, Delta, Lambda, a_j) >= 0)
            {
//                std::cout << "lambda_min <= 0 && h(- lambda_min) >= 0" << std::endl;//---
//                std::cout << "t2" << std::endl;//----
                sigmaStar = -lambda_min;
                realdp *v = new realdp[k + 1];
                for (integer i = 0; i < k + 1; i++)
                {
                    if (fabs(Lambdaptr[i] + sigmaStar) > tolLocalNewton)
                    {
                        v[i] = a_jptr[i] / (Lambdaptr[i] + sigmaStar);
                    }
                    else
                    {
                        v[i] = 0;
                    }
                }

//                P_parallel.Print("P_parallel 1:");//---
                const realdp *P_parallelptr = P_parallel.ObtainReadData();
                gemm_(GLOBAL::N, GLOBAL::N, &n, &GLOBAL::IONE, &k, &GLOBAL::DNONE, const_cast<realdp *> (P_parallelptr), &n, v, &k, &GLOBAL::DZERO, pStar, &n);
                delete[] v;
                if (fabs(gamma + sigmaStar) > tolLocalNewton)
                {
//                    std::cout << "t3" << std::endl;//----
                    eta2 = (P_parallel * g_parallel - gf1) / (gamma + sigmaStar) + eta2;
                }

                if (lambda_min < 0)
                {
//                    std::cout << "t4" << std::endl;//----
                    realdp alpha = sqrt(Delta * Delta - dot_(&n, pStar, &GLOBAL::IONE, pStar, &GLOBAL::IONE));

                    Vector zstar(n);
                    if (fabs(lambda_min - Lambdaptr[0]) < tolLocalNewton)
                    {
//                        std::cout << "t5" << std::endl;//----
                        Vector P_parallel_firstcol = P_parallel.GetSubmatrix(0, n - 1, 0, 0);
                        zstar = (alpha / std::sqrt(P_parallel_firstcol.DotProduct(P_parallel_firstcol))) * P_parallel_firstcol;
//                        P_parallel.Print("P_parallel 2:");//---
                    }
                    else
                    {
//                        std::cout << "t6" << std::endl;//----
                        realdp norm_umin = 1;
                        for (integer i = 0; i < k; i++)
                        {
                            Vector subM = P_parallel.GetSubmatrix(i, i, 0, k - 1);
                            zstar.AlphaABaddBetaThis(1, P_parallel, GLOBAL::N, subM, GLOBAL::T, 0); /*zstar = P_parallel * P_parallel.GetSubmatrix(i, i, 0, k - 1).GetTranspose();*/
                            realdp *zstarptr = zstar.ObtainWritePartialData();
                            zstarptr[i] += 1;
                            norm_umin = std::sqrt(zstar.DotProduct(zstar));
                            if (norm_umin > tolLocalNewton)
                                break;
                        }
                        zstar = (alpha / norm_umin) * zstar;
                    }
                    eta2 = zstar + eta2;
                }
                tCGstatusSM = TRSM_NEGCURVTURE;
            }
            else
            {
//                std::cout << "lambda_min > 0 || h(- lambda_min) < 0" << std::endl;//---
//                std::cout << "t7" << std::endl;//----
                if (lambda_min > 0)
                {
//                    std::cout << "t8" << std::endl;//----
                    sigmaStar = LocalNewton(0, 2 * LengthSY + 10, tolLocalNewton, Lambda, a_j);
                }
                else
                {
//                    std::cout << "t9" << std::endl;//----
                    realdp sigmaHat = 0;
                    for (integer i = 0; i < k + 1; i++)
                    {
                        if (sigmaHat < a_jptr[i] / Delta - Lambdaptr[i])
                            sigmaHat = a_jptr[i] / Delta - Lambdaptr[i];
                    }
                    if (sigmaHat > -lambda_min)
                    {
//                        std::cout << "t10" << std::endl;//----
                        sigmaStar = LocalNewton(sigmaHat, 2 * LengthSY + 10, tolLocalNewton, Lambda, a_j);
                    }
                    else
                    {
                        sigmaStar = LocalNewton(-lambda_min, 2 * LengthSY + 10, tolLocalNewton, Lambda, a_j);
//                        std::cout << "t11, sigmaStar:" << sigmaStar << std::endl;//----
                    }
                }
                if(k == 0)
                {
//                    std::cout << "t12" << std::endl;//----
                    ComputeSBySMW(gamma + sigmaStar, Psig, Vector (), PMGQ, Psi);
                }
                else
                {
//                    std::cout << "t13" << std::endl;//----
                    ComputeSBySMW(gamma + sigmaStar, Psig, R, PMGQ, Psi); /*gf1, Psig, YMGS, PMGQ, RT*/
                }
                tCGstatusSM = TRSM_EXCREGION;
            }
//        printf("neta2:%e\n", std::sqrt(Mani->Metric(x1, eta2, eta2)));//---
        Mani->Projection(x1, eta2, &eta2);
        /*recover the types of data*/
        gf1.Setiscomplex(gf1iscomplex);
        eta2.Setiscomplex(gf1iscomplex);
        
//        printf("neta2:%e\n", std::sqrt(Mani->Metric(x1, eta2, eta2)));//---
//        eta2.Print("eta2:");//----
        
        Mani->VectorLinearCombination(x1, - sigmaStar, eta2, -1, gf1, &Heta2); /* Heta2 = - sigmaStar * eta2 - gf1; */

//        std::cout << "Delta:" << Delta << ", norm etax2:" << std::sqrt( Mani->Metric(x1, eta2, eta2) ) << std::endl;//--
//        if( std::fabs(Delta - std::sqrt( Mani->Metric(x1, eta2, eta2) )) > 1e-5)
//            std::cout << "warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;//----
    };

    void LRTRSR1::ComputeSBySMW(realdp tauStar, const Vector &Psig, const Vector &R, const Vector &PMGQ, const Vector &Psi)
    {/*gamma, gf1, Psig, YMGS, PMGQ, R*/
        if (Currentlength != 0)
        {
//            realdp eps = 100 * std::numeric_limits<realdp>::epsilon();
            Vector vw(PMGQ);
            vw.AlphaABaddBetaThis(tauStar, R, GLOBAL::T, R, GLOBAL::N, tauStar * tauStar);/*Vector vw = tauStar * (tauStar * PMGQ + (R.GetTranspose() * R));*/
            
            /*if vw is a rank deficient matrix, then eta2 = (static_cast<realdp> (-1) / tauStar) * gf1; */
            vw.HHRDecom();
            const realdp *HHRptr = vw.Field("_HHR").ObtainReadData();
            realdp minv = std::abs(HHRptr[0]), maxv = std::abs(HHRptr[0]);
            for(integer i = 0; i < vw.Getrow(); i++)
            {
                if(minv > std::abs(HHRptr[i + vw.Getrow() * i]))
                    minv = std::abs(HHRptr[i + vw.Getrow() * i]);
                if(maxv < std::abs(HHRptr[i + vw.Getrow() * i]))
                    maxv = std::abs(HHRptr[i + vw.Getrow() * i]);
            }
//            printf("minv:%e, maxv:%e\n", minv, maxv);//----
            
            if(minv / maxv < 1e-9 || maxv == static_cast<realdp> (0)) /*rank deficient*/
//            if(minv / maxv < eps || maxv == static_cast<realdp> (0)) /*rank deficient*/
            {
//                printf("h1\n");//---
                eta2 = gf1; eta2.ScalarTimesThis(static_cast<realdp> (-1) / tauStar); /* eta2 = (static_cast<realdp> (-1) / tauStar) * gf1; */
                return;
            }
//            printf("h2, tauStar:%e\n", tauStar);//---
//            vw.Print("vw:");//---

            /*otherwise, use below eta2*/
            /* eta2 = Psi * (vw % Psig) + (static_cast<realdp> (-1) / tauStar) * gf1; */
            Vector tmp = vw % Psig;
            eta2 = gf1;
            eta2.AlphaABaddBetaThis(1, Psi, GLOBAL::N, tmp, GLOBAL::N, static_cast<realdp> (-1) / tauStar);
            
            return;
        }
        /*eta2 = gf1 * (static_cast<realdp> (-1) / tauStar);*/
        eta2 = gf1;
        eta2.ScalarTimesThis(static_cast<realdp> (-1) / tauStar);
    };

    realdp LRTRSR1::PhiBar_fg(realdp nlambda_min, realdp Delta, const Vector &Lambda, const Vector &a_j, realdp *gf)
    {
        integer k = Currentlength;    /*number of columns*/
        Vector tmp = Lambda + nlambda_min;
        
        realdp eps_tol = static_cast<realdp> (1e-10);
        realdp gradf = 0;
        const realdp *a_jptr = a_j.ObtainReadData();
        realdp *tmpptr = tmp.ObtainWritePartialData();

        bool flag = false;
        for (integer i = 0; i < k + 1; i++)
        {
            if (fabs(a_jptr[i]) < eps_tol || fabs(tmpptr[i]) < eps_tol)
            {
                flag = true;
                break;
            }
        }
        
        if (flag)
        {
            realdp pnorm2 = 0;
            for (integer i = 0; i < k + 1; i++)
            {
                if (fabs(a_jptr[i]) > eps_tol && fabs(tmpptr[i]) < eps_tol)
                {
                    gradf = static_cast<realdp> (1) / eps_tol;
                    *gf = gradf;
                    return static_cast<realdp> (-1) / Delta;
                }
                else
                if (fabs(a_jptr[i]) > eps_tol && fabs(tmpptr[i]) > eps_tol)
                {
                    pnorm2 += a_jptr[i] * a_jptr[i] / tmpptr[i] / tmpptr[i];
                    gradf += a_jptr[i] * a_jptr[i] / tmpptr[i] / tmpptr[i] / tmpptr[i];
                }
            }
            realdp norm = sqrt(pnorm2);
            gradf = gradf / norm / norm / norm;
            *gf = gradf;
            return static_cast<realdp> (1) / norm - static_cast<realdp> (1) / Delta;
        }
        
        for (integer i = 0; i < k + 1; i++)
            gradf += a_jptr[i] * a_jptr[i] / tmpptr[i] / tmpptr[i] / tmpptr[i];
        
        for (integer i = 0; i < k + 1; i++)
            tmpptr[i] = a_jptr[i] / tmpptr[i];
        
        realdp norm = tmp.Fnorm();
        
        gradf = gradf / norm / norm / norm;
        *gf = gradf;
        return static_cast<realdp> (1) / norm - static_cast<realdp> (1) / Delta;
    };

    realdp LRTRSR1::PhiBar_fg(realdp nlambda_min, realdp Delta, const Vector &Lambda, const Vector &a_j)
    {
        integer k = Currentlength;    /*number of columns*/
        Vector tmp = Lambda + nlambda_min;
        
        realdp eps_tol = static_cast<realdp> (1e-10);
        const realdp *a_jptr = a_j.ObtainReadData();
        realdp *tmpptr = tmp.ObtainWritePartialData();

        bool flag = false;
        for (integer i = 0; i < k + 1; i++)
        {
            if (fabs(a_jptr[i]) < eps_tol || fabs(tmpptr[i]) < eps_tol)
            {
                flag = true;
                break;
            }
        }
        
        if (flag)
        {
            realdp pnorm2 = 0;
            for (integer i = 0; i < k + 1; i++)
            {
                if (fabs(a_jptr[i]) > eps_tol && fabs(tmpptr[i]) < eps_tol)
                {
                    return static_cast<realdp> (-1) / Delta;
                }
                else
                if (fabs(a_jptr[i]) > eps_tol && fabs(tmpptr[i]) > eps_tol)
                {
                    pnorm2 += a_jptr[i] * a_jptr[i] / tmpptr[i] / tmpptr[i];
                }
            }
            realdp norm = sqrt(pnorm2);
            return static_cast<realdp> (1) / norm - static_cast<realdp> (1) / Delta;
        }
        for (integer i = 0; i < k + 1; i++)
            tmpptr[i] = a_jptr[i] / tmpptr[i];
        realdp norm = tmp.Fnorm();
        return static_cast<realdp> (1) / norm - static_cast<realdp> (1) / Delta;
    };

    realdp LRTRSR1::LocalNewton(realdp x0, integer maxIter, realdp tol, const Vector &Lambda, const Vector &a_j)
    {
        integer k = 0;
        realdp gf = 0;
        realdp fv = PhiBar_fg(x0, Delta, Lambda, a_j, &gf);
        while (fabs(fv) > 100 * std::numeric_limits<realdp>::epsilon() && k < maxIter)
        {
            x0 -= fv / gf;
            fv = PhiBar_fg(x0, Delta, Lambda, a_j, &gf);
            k++;
        }
//        std::cout << "k:" << k << ", fv in LocalNewton:" << fv << std::endl;//---
        return x0;
    };

    void LRTRSR1::CheckParams(void)
    {
        SolversSMTR::CheckParams();
        char YES[] = "YES";
        char NO[] = "NO";
        char *status;

        printf("LRTRSR1 METHOD PARAMETERS:\n");
        status = (LengthSY >= 0) ? YES : NO;
        printf("LengthSY      :%15d[%s],\t", LengthSY, status);
        status = (Lgamma > 0) ? YES : NO;
        printf("Lgamma        :%15g[%s],\n", Lgamma, status);
    };

    Vector &LRTRSR1::HessianEta(const Vector &Eta, Vector *result)
    {
        return HvLRTRSR1(Eta, result);
    };

    void LRTRSR1::UpdateData(void)
    {
        UpdateDataLRTRSR1();
    };

    void LRTRSR1::Acceptence(void)
    {
        for (integer i = 0; i < Currentlength; i++)
        {
            Mani->VectorTransport(x1, eta2, x2, YMGS[i], &YMGS[i]);
        }
    };

    void LRTRSR1::PrintInfo(void)
    {
        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));

        printf("|gf|:%.3e,time:%.2g,", ngf2, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);

        printf("rho:%.2e,radius:%.3e,tCGstatus:%s,innerIter:%d,", rho, Delta, tCGstatusSetSMnames[tCGstatusSM].c_str(), innerIter);
        
        printf("gamma:%.3e,inpss:%.3e,inpsy:%.3e,inpyy:%.3e,IsUpdateHessian:%d,", gamma, inpss, inpsy, inpyy, isupdated);
        
        printf("nf:%d,ng:%d,", nf, ng);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
    };

    Vector &LRTRSR1::HvLRTRSR1(const Vector &Eta, Vector *result)
    {
        /* In LRTRSR1, only B_k * s_k is computed, this can be done by exploiting the optimality condition
         of the subproblem. Heta2 has been computed in tCG_TR and therefore do not need be recomputed. */
        return *result;
        
//        /* This function makes use of SS, SY and gamma to evaluate the action of Hessian approximation [HAG2014, (64)].
//        [HAG2014]: W. Huang, P.-A. Absil, and K. A. Gallivan. A Riemannian symmetric rank-one trustregion method.
//        Mathematical Programming, 150(2):179?16, February 2015.
//
//        SS is the Q in (46), SY is the P in (46), PMGQ is the P - gamma Q in (46).
//        */
//        Vector v(Currentlength), v2;
//        realdp *vptr = v.ObtainWriteEntireData();
//
//        for (integer i = 0; i < Currentlength; i++)
//            vptr[i] = Mani->Metric(x1, YMGS[i], Eta);
//
//        if (Currentlength > 0)
//        {
//            v2 = PMGQ % v;
//        }
//        const realdp *v2ptr = v2.ObtainReadData();
//
//        Mani->ScalarTimesVector(x1, gamma, Eta, result);
//        for (integer i = 0; i < Currentlength; i++)
//        {
//            Mani->ScalarVectorAddVector(x1, v2ptr[i], YMGS[i], *result, result);
//        }
//        return *result;
    };

    void LRTRSR1::UpdateDataLRTRSR1(void)
    {
        realdp denorminator, norm2ymBs;
        realdp mintolsq = std::numeric_limits<realdp>::epsilon();
        realdp mintol = sqrt(mintolsq);
        Prob->Grad(x2, &gf2); ng++;
        s = eta2;
        Mani->InverseVectorTransport(x1, eta2, x2, gf2, &eta1); nV++;
        Mani->VectorLinearCombination(x1, 1, eta1, -1, gf1, &y);
        
        if (iter == 0) /* This is for the robustness when the cost function is quadratic and its Hessian is identity everywhere.*/
        {
            inpss = Mani->Metric(x1, s, s);
            inpsy = Mani->Metric(x1, s, y);
            inpyy = Mani->Metric(x1, y, y);
            if(std::fabs(inpsy) < std::numeric_limits<realdp>::epsilon())
                gamma = 0;
            else
                gamma = (inpyy / std::fabs(inpsy) > Lgamma * std::fabs(inpsy) / inpss) ? Lgamma * std::fabs(inpsy) / inpss : inpyy / std::fabs(inpsy);
            
            gamma = (inpsy > 0) ? gamma : - gamma;
//            gamma = inpsy / inpss;;
            Mani->ScalarTimesVector(x1, gamma, s, &Heta2);
        }
        
        Vector zeta(y); Mani->ScalarVectorAddVector(x1, -1, Heta2, y, &zeta);
        denorminator = Mani->Metric(x1, s, zeta);
        norm2ymBs = Mani->Metric(x1, zeta, zeta);
        inpss = Mani->Metric(x1, s, s);
        inpsy = Mani->Metric(x1, s, y);
        
//        std::cout << (denorminator * denorminator >= mintolsq * inpss * norm2ymBs) << ":" << (norm2ymBs >= mintolsq || ngf2 / ngf0 < 1e-3) << ":" << (iter != 0 || fabs(gamma - inpsy / inpss) > mintol) << std::endl;//---
        if (denorminator * denorminator >= mintol * inpss * norm2ymBs //--&& (norm2ymBs >= mintolsq || ngf2 / ngf0 < 1e-3)
            && (iter != 0 || fabs(gamma - inpsy / inpss) > mintol)) /* This is for the robustness when the cost
             function is quadratic and its Hessian is identity everywhere. */
        {
            inpyy = Mani->Metric(x1, y, y);
            
//            if(Currentlength >= LengthSY)
//                Currentlength = 0;
//            if(isupdated && Currentlength == 0)
//                gamma = inpsy / inpss;
            
            if(Currentlength >= LengthSY)
            {
                gamma = (inpyy / std::fabs(inpsy) > Lgamma * std::fabs(inpsy) / inpss) ? Lgamma * std::fabs(inpsy) / inpss : inpyy / std::fabs(inpsy);
                gamma = (inpsy > 0) ? gamma : - gamma;
                Currentlength = 0;
                PsiTPsi = Vector();
                PMGQ = Vector();
            }
            
            /*if s and y are accepted, then S and Y need to be updated. It follows that the matrices SY and SS need to be update.*/
            if (Currentlength < LengthSY)
            {
                Mani->ScalarVectorAddVector(x1, - gamma, s, y, &YMGS[Currentlength]); /* YMGS[Currentlength] = y - gamma * s */
                Vector tmp = PsiTPsi;
                PsiTPsi = Vector(Currentlength + 1, Currentlength + 1);
                PsiTPsi.SubmatrixAssignment(0, Currentlength - 1, 0, Currentlength - 1, tmp);
                realdp *PsiTPsiptr = PsiTPsi.ObtainWritePartialData();
                for(integer i = 0; i < Currentlength + 1; i++)
                {
                    PsiTPsiptr[i + Currentlength * (Currentlength + 1)] = Mani->Metric(x1, YMGS[i], YMGS[Currentlength]);
                    PsiTPsiptr[Currentlength + i * (Currentlength + 1)] = PsiTPsiptr[i + Currentlength * (Currentlength + 1)];
                }
                
                tmp = PMGQ;
                PMGQ = Vector(Currentlength + 1, Currentlength + 1);
                PMGQ.SubmatrixAssignment(0, Currentlength - 1, 0, Currentlength - 1, tmp);
                realdp *PMGQptr = PMGQ.ObtainWritePartialData();
                for(integer i = 0; i < Currentlength + 1; i++)
                {
                    PMGQptr[i + Currentlength * (Currentlength + 1)] = Mani->Metric(x1, s, YMGS[i]);
                    PMGQptr[Currentlength + i * (Currentlength + 1)] = PMGQptr[i + Currentlength * (Currentlength + 1)];
                }
                Currentlength++;
            }

            isupdated = true;
        }
        else
        {
            isupdated = false;
        }
    };

}; /*end of ROPTLIB namespace*/
