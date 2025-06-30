
#include "Solvers/LRBroydenFamily.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    LRBroydenFamily::LRBroydenFamily(const Problem *prob, const Variable *initialx)
    {
        Initialization(prob, initialx);
    };
    
    void LRBroydenFamily::SetProbX(const Problem *prob, const Variable *initialx)
    {
        SolversSMLS::SetProbX(prob, initialx);
        prob->SetUseGrad(true);
        prob->SetUseHess(false);
        //prob->SetUseHess(true);
        s = Prob->GetDomain()->GetEMPTY();
        y = Prob->GetDomain()->GetEMPTY();
        Py = Prob->GetDomain()->GetEMPTY();
    };
    
    void LRBroydenFamily::SetDefaultParams(void)
    {
        SolversSMLS::SetDefaultParams();
        isconvex = false;
        nu = static_cast<realdp> (1e-4);
        mu = 1;
        LengthSY = 4;
        phiset = 0;
        LMrestart = false;
        S = nullptr;
        Y = nullptr;
        SY = nullptr;
        SS = nullptr;
        YY = nullptr;
        store_phi = nullptr;
        store_sr = nullptr;
        phi = 0;
        Currentlength = 0;
        beginidx = 0;
        gamma = 1;
        delta = 1.5;
        InitSteptype = LSSM_QUADINTMOD;
        SolverName.assign("LRBroydenFamily");
    };
    
    void LRBroydenFamily::SetParams(PARAMSMAP params)
    {
        SolversSMLS::SetParams(params);
        PARAMSMAP::iterator iter;
        for (iter = params.begin(); iter != params.end(); iter++)
        {
            if (iter->first == static_cast<std::string> ("isconvex"))
            {
                isconvex = ((static_cast<integer> (iter->second)) != 0);
            }
            else
                if (iter->first == static_cast<std::string> ("LengthSY"))
                {
                    LengthSY = static_cast<integer> (iter->second);
                }
                else
                    if (iter->first == static_cast<std::string> ("LMrestart"))
                    {
                        LMrestart = static_cast<integer> (iter->second);
                    }
                    else
                        if (iter->first == static_cast<std::string> ("nu"))
                        {
                            nu = iter->second;
                        }
                        else
                            if (iter->first == static_cast<std::string> ("mu"))
                            {
                                mu = iter->second;
                            }
                            else
                                if (iter->first == static_cast<std::string> ("phiset"))
                                {
                                    phiset = static_cast<double> (iter->second);
                                }
        }
    };
    
    LRBroydenFamily::~LRBroydenFamily(void)
    {
        DeleteVectors(S, LengthSY);
        DeleteVectors(Y, LengthSY);
        if (store_phi != nullptr)
            delete[] store_phi;
        if (store_sr != nullptr)
            delete[] store_sr;
        if (SY != nullptr)
            delete[] SY;
        if (SS != nullptr)
            delete[] SS;
        if (YY != nullptr)
            delete[] YY;
    };
    
    void LRBroydenFamily::Run(void)
    {
        DeleteVectors(S, LengthSY);
        NewVectors(S, LengthSY);
        DeleteVectors(Y, LengthSY);
        NewVectors(Y, LengthSY);
        if (store_phi != nullptr)
            delete[] store_phi;
        if (store_sr != nullptr)
            delete[] store_sr;
        if (SY != nullptr)
            delete[] SY;
        if (SS != nullptr)
            delete[] SS;
        if (YY != nullptr)
            delete[] YY;
        store_phi = new realdp[LengthSY];
        store_sr = new integer[LengthSY];
        SY = new realdp[LengthSY*LengthSY];
        SS = new realdp[LengthSY*LengthSY];
        YY = new realdp[LengthSY*LengthSY];
        SolversSMLS::Run();
    };
    
    void LRBroydenFamily::CheckParams(void)
    {
        SolversSMLS::CheckParams();
        char YES[] = "YES";
        char NO[] = "NO";
        char *status;
        
        printf("LRBroydenFamily METHOD PARAMETERS:\n");
        status = (nu >= 0 && nu < 1) ? YES : NO;
        printf("nu            :%15g[%s],\t", nu, status);
        status = (mu >= 0) ? YES : NO;
        printf("mu            :%15g[%s],\n", mu, status);
        status = YES;
        printf("isconvex      :%15d[%s],\t", isconvex, status);
        status = (LengthSY >= 0) ? YES : NO;
        printf("LengthSY      :%15d[%s],\n", LengthSY, status);
        status = YES;
        printf("LMrestart     :%15d[%s],\t", LMrestart, status);
        status = (phiset >= 0) ? YES : NO;;
        printf("phiset        :%15g[%s],\n", phiset, status);
    };
    
    void LRBroydenFamily::GetSearchDir(void)
    {
        HvLRBroydenFamily(gf1, &eta1);
        Mani->ScalarTimesVector(x1, -1.0, eta1, &eta1);
    };
    
    void LRBroydenFamily::UpdateData(void)
    {
        UpdateDataLRBroydenFamily();
    };
    
    void LRBroydenFamily::PrintInfo(void)
    {
        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));
        
        printf("|gf|:%.3e,t0:%.2e,t:%.2e,s0:%.2e,s:%.2e,time:%.2g,", ngf2, initiallength, stepsize, initialslope, newslope, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        
        printf("\n\tbetay:%.3e,gamma:%.3e,inpss:%.3e,inpsy:%.3e,inpyy:%.3e,IsUpdateHessian:%d,", betay, gamma, inpss, inpsy, inpyy, isupdated);
        
        printf("nf:%d,ng:%d,", nf, ng);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
    };
    
    Vector &LRBroydenFamily::HvLRBroydenFamily(const Vector &v, Vector *result)
    {
        integer idx;
        *result = v;
        
        if (Currentlength == 0)
        {
            Prob->PreConditioner(x1, *result, &Py);
            Mani->ScalarTimesVector(x1, gamma, Py, result);
            return *result;
        }
        
        realdp sBs,yHy,yTs,phic,Bphi; //Bphi: \Phi
        realdp alpha_hat,alpha_tilde,beta_hat,beta_tilde,delta_hat,delta_tilde;
        realdp ppsis,ppsiy;
        integer *ifSR1 = new integer[Currentlength];
        realdp *M_hat = new realdp[4*Currentlength*Currentlength];
        realdp *M_tilde = new realdp[4*Currentlength*Currentlength];
        realdp *psis = new realdp[2*Currentlength]; //Psi'*s
        realdp *psiy = new realdp[2*Currentlength]; //Psi_tilde'*y
        realdp *p = new realdp[2*Currentlength]; //M_hat*(Psi'*s)
        realdp *p_tilde = new realdp[2*Currentlength]; //M_tilde*(Psi_tilde'*y)
        realdp *psiv = new realdp[2*Currentlength]; //Psi_tilde'*v
        realdp *Mpsiv = new realdp[2*Currentlength]; //M_tilde*(Psi_tilde'*v)
        realdp apb, bcube, kbfgs, koc; //condition number for hybrid Davidon, kbfgs denotes kappa(BFGS), koc denotes kappa(Davidon)
        /* initial  */
        sBs = SS[beginidx*LengthSY + beginidx]/gamma;
        yHy = YY[beginidx*LengthSY + beginidx]*gamma;
        yTs = SY[beginidx*LengthSY + beginidx];
        phic = (yTs*yTs)/(yTs*yTs-sBs*yHy);
        //------------------------------ Davidon's phi -------------------------------------
        apb = yHy + yTs;
        bcube = yTs * yTs * yTs;
        kbfgs = (apb*apb*sBs - 2*bcube + apb*sqrt(apb*apb*sBs*sBs-4*bcube*sBs))/(2*bcube);
        if (yTs <= static_cast<realdp> (2) *sBs*yHy/(sBs+yHy))
        {
//            std::cout << "t1" << std::endl;//---
            phi = yTs*(yHy-yTs)/(sBs*yHy-yTs*yTs);
            ifSR1[0] = 0;
            koc = (2*yHy*yTs - yTs*yTs + 2*sqrt(yHy*yHy*sBs*sBs-yHy*yTs*yTs*sBs))/(yTs*yTs); //condition number of Davidon's update
        }
        else
        {
//            std::cout << "t2" << std::endl;//---
            phi = yTs/(yTs-sBs);
            ifSR1[0] = 1;
            if (yHy > yTs)
            {
                koc = (yHy - yTs)/(yTs - sBs);
            }
            else
            {
                koc = (sBs - yTs)/(yTs - yHy);
            }
        }
//        std::cout << "phi:" << phi << std::endl;//--
        //-------------------perturbation---------------
        /*if (phiset == 31415)
        {
            phi = 0;
            ifSR1[0] = 0;      
        }
        double eps = rand();
        if (phiset != 0)
        {
            phi = phi + 0.1 * (eps / RAND_MAX - 0.5) * 2;
        }
        if (phiset != 0 &&  phi <= 0.85*phic )
        {
            phi = 0;
            ifSR1[0] = 0;
        } */
        /* Historical */
        if( (phiset == 1 || phiset == 2) && Currentlength > 1)
        {
            phi = store_phi[(beginidx + 1) % LengthSY];
            ifSR1[0] = store_sr[(beginidx + 1) % LengthSY];
            if (phiset == 2 && phi <= 0.95*phic)
            {
                phi = 0;
                ifSR1[0] = 0;
            }
            if (phiset == 2 && phi >= 0.95)
            {
                phi = 0;
                ifSR1[0] = 0;
            }
        }
        else if (phiset == 1 || phiset == 2)
        {
            sr_value = ifSR1[0];
        }
        // ------------------hybrid------------------------
        else
        {
//            if (kbfgs < phiset * koc)
            if (kbfgs < koc * delta)
            {
                phi = 0;
                ifSR1[0] = 0;
            }
        }
//        std::cout << "phi:" << phi << ", sBs:" << sBs << ",yHy:" << yHy << ", all:" << sBs*yHy/(yTs*yTs) << std::endl;//--
//        std::cout << "deno:" << static_cast<realdp> (1) - phi+phi*sBs*yHy/(yTs*yTs) << std::endl;//--
        Bphi = (static_cast<realdp> (1) - phi) / (static_cast<realdp> (1) - phi+phi*sBs*yHy/(yTs*yTs));
        
        if (std::isnan(Bphi) || std::isinf(Bphi)) /*reset to BFGS when it is nan or inf*/
        {
            phi = 0;
            Bphi = (static_cast<realdp> (1) - phi) / (static_cast<realdp> (1) - phi+phi*sBs*yHy/(yTs*yTs));
        }
        
        //phi_ave += phi;
        //std::cout << phi << std::endl;
        if (ifSR1[0] == 1)
        {
//            std::cout << "s1" << std::endl;//---
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
//            std::cout << "Bphi:" << Bphi << ", yTs:" << yTs << std::endl;//---
            M_hat[0] = -beta_hat;
            M_hat[1] = static_cast<realdp> (0);
            M_hat[2] = static_cast<realdp> (0);
            M_hat[3] = static_cast<realdp> (0);
            M_tilde[0] = -beta_tilde;
            M_tilde[1] = static_cast<realdp> (0);
            M_tilde[2] = static_cast<realdp> (0);
            M_tilde[3] = static_cast<realdp> (0);
        }
        else
        {
//            std::cout << "s2" << std::endl;//---
            alpha_hat = (phi - static_cast<realdp> (1)) / sBs;
            alpha_tilde = (static_cast<realdp> (1) + Bphi * yHy / yTs) / yTs;
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
            delta_hat = (static_cast<realdp> (1) + phi * sBs / yTs) / yTs;
            delta_tilde = (Bphi - static_cast<realdp> (1)) / yHy;
            M_hat[0] = alpha_hat;
            M_hat[1] = beta_hat;
            M_hat[2] = beta_hat;
            M_hat[3] = delta_hat;
            M_tilde[0] = alpha_tilde;
            M_tilde[1] = beta_tilde;
            M_tilde[2] = beta_tilde;
            M_tilde[3] = delta_tilde;
        }
        
//        std::cout << "tttest0:" << M_tilde[0] << std::endl;//---
        /* loop */
        for (integer i = 1; i < Currentlength; i++)
        {
            idx = (beginidx + i) % LengthSY;
            for (integer j = 0; j < i; j++)
            {
                integer jdx = (beginidx + j) % LengthSY;
                psis[2*j] = SS[idx * LengthSY + jdx]/gamma;
                psis[2*j+1] = SY[idx * LengthSY + jdx];
                psiy[2*j] = SY[jdx * LengthSY + idx];
                psiy[2*j+1] = YY[idx * LengthSY + jdx]*gamma;
                if (ifSR1[j] == 1)
                {
                    psis[2*j] = psis[2*j+1] - psis[2*j];
                    psis[2*j+1] = static_cast<realdp> (0);
                    psiy[2*j] = psiy[2*j+1] - psiy[2*j];
                    psiy[2*j+1] = static_cast<realdp> (0);
                }
            }
            for (integer j = 0; j < 2*i; j++)
            {
                p[j] = static_cast<realdp> (0);
                p_tilde[j] = static_cast<realdp> (0);
                for (integer k = 0; k < 2*i; k++)
                {
                    p[j] = p[j] + M_hat[j*(2*i)+k] * psis[k];
                    p_tilde[j] = p_tilde[j] + M_tilde[j*(2*i)+k] * psiy[k];
                }
            }
            ppsis = static_cast<realdp> (0);
            ppsiy = static_cast<realdp> (0);
            for (integer k = 0; k < 2*i; k++)
            {
                ppsis = ppsis + p[k]*psis[k];
                ppsiy = ppsiy + p_tilde[k]*psiy[k];
            }
            sBs = SS[idx * LengthSY + idx] / gamma + ppsis;
            yHy = YY[idx * LengthSY + idx] * gamma + ppsiy;
            yTs = SY[idx * LengthSY + idx];
            phic = (yTs*yTs)/(yTs*yTs-sBs*yHy);
            //---------------------------------------------------------------
            apb = yHy + yTs;
            bcube = yTs * yTs * yTs;
            kbfgs = (apb*apb*sBs - 2*bcube + apb*sqrt(apb*apb*sBs*sBs-4*bcube*sBs))/(2*bcube);
            if (yTs <= static_cast<realdp> (2) *sBs*yHy/(sBs+yHy))
            {
                phi = yTs*(yHy-yTs)/(sBs*yHy-yTs*yTs);
                ifSR1[i] = 0;
                koc = (2*yHy*yTs - yTs*yTs + 2*sqrt(yHy*yHy*sBs*sBs-yHy*yTs*yTs*sBs))/(yTs*yTs);
            }
            else
            {
                phi = yTs/(yTs-sBs);
                ifSR1[i] = 1;
                if (yHy > yTs)
                {
                    koc = (yHy - yTs)/(yTs - sBs);
                }
                else
                {
                    koc = (sBs - yTs)/(yTs - yHy);
                }
            }
            
            if ( (phiset == 1 || phiset == 2) && i < Currentlength - 1)
            {
                phi = store_phi[(beginidx + 1 + i) % LengthSY];
                ifSR1[i] = store_sr[(beginidx + 1 + i) % LengthSY];
                if (phiset == 2 && phi <= 0.95*phic)
                {
                    phi = 0;
//                    phi_ave++;
                    ifSR1[i] = 0;
                }
                if (phiset == 2 && phi >= 0.95)
                {
                    phi = 0;
//                    phi_ave++;
                    ifSR1[i] = 0;
                }
            }
            else if (phiset == 1 || phiset == 2)
            {
                sr_value = ifSR1[i];
            }
            // ------------------hybrid------------------------
            else
            {
//                if (kbfgs < phiset * koc)
                if (kbfgs < koc * delta)
                {
                    phi = 0;
                    ifSR1[0] = 0;
                }
            }
            //phi_ave += phi;PrintInfo
            //std::cout << phi << std::endl;
            //----------------------perturbation----------
            /*if (phiset == 31415)
            {
                phi = 0;
                ifSR1[i] = 0;      
            }
            double eps = rand();
            if (phiset != 0)
            {
                phi = phi + 0.1 * (eps / RAND_MAX - 0.5) * 2;
            }
            if (phiset != 0 && phi <= 0.85*phic )
            {
                phi = 0;
                ifSR1[i] = 0;
            }*/
            Bphi = (static_cast<realdp> (1) - phi) / (static_cast<realdp> (1) - phi+phi*sBs*yHy/(yTs*yTs));
            
            if (std::isnan(Bphi) || std::isinf(Bphi)) /*reset to BFGS when it is nan or inf*/
            {
                phi = 0;
                Bphi = (static_cast<realdp> (1) - phi) / (static_cast<realdp> (1) - phi+phi*sBs*yHy/(yTs*yTs));
            }
            
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
            if (ifSR1[i] == 0)
            {
                alpha_hat = (phi - static_cast<realdp> (1)) / sBs;
                alpha_tilde = (static_cast<realdp> (1) + Bphi * yHy / yTs) / yTs;
                delta_hat = (static_cast<realdp> (1) + phi * sBs / yTs) / yTs;
                delta_tilde = (Bphi - static_cast<realdp> (1)) / yHy;
                for (integer k = 2*i-1; k >= 0; k--)
                {
                    for (integer j = 2*i-1; j >= 0; j--)
                    {
                        M_hat[k*(2*i+2)+j] = M_hat[k*(2*i)+j] + alpha_hat*p[j]*p[k];
                        M_tilde[k*(2*i+2)+j] = M_tilde[k*(2*i)+j]+delta_tilde*p_tilde[j]*p_tilde[k];
                    }
                }
                for (integer k = 0; k < 2*i; k++)
                {
                    M_hat[(k+1)*(2*i+2)-2] = alpha_hat*p[k];
                    M_hat[(k+1)*(2*i+2)-1] = beta_hat*p[k];
                    M_tilde[(k+1)*(2*i+2)-2] = beta_tilde*p_tilde[k];
                    M_tilde[(k+1)*(2*i+2)-1] = delta_tilde*p_tilde[k];
                }
                for (integer j = 0; j < 2*i; j++)
                {
                    M_hat[(2*i)*(2*i+2)+j] = alpha_hat*p[j];
                    M_hat[(2*i+1)*(2*i+2)+j] = beta_hat*p[j];
                    M_tilde[(2*i)*(2*i+2)+j] = beta_tilde*p_tilde[j];
                    M_tilde[(2*i+1)*(2*i+2)+j] = delta_tilde*p_tilde[j];
                }
                M_hat[(2*i+1)*(2*i+2)-2] = alpha_hat;
                M_hat[(2*i+1)*(2*i+2)-1] = beta_hat;
                M_hat[(2*i+2)*(2*i+2)-2] = beta_hat;
                M_hat[(2*i+2)*(2*i+2)-1] = delta_hat;
                M_tilde[(2*i+1)*(2*i+2)-2] = alpha_tilde;
                M_tilde[(2*i+1)*(2*i+2)-1] = beta_tilde;
                M_tilde[(2*i+2)*(2*i+2)-2] = beta_tilde;
                M_tilde[(2*i+2)*(2*i+2)-1] = delta_tilde;
            }
            else
            {
                for (integer k = 2*i-1; k >= 0; k--)
                {
                    for (integer j = 2*i-1; j >= 0; j--)
                    {
                        M_hat[k*(2*i+2)+j] = M_hat[k*(2*i)+j] - beta_hat*p[j]*p[k];
                        M_tilde[k*(2*i+2)+j] = M_tilde[k*(2*i)+j]-beta_tilde*p_tilde[j]*p_tilde[k];
                    }
                }
                for (integer k = 0; k < 2*i; k++)
                {
                    M_hat[(k+1)*(2*i+2)-2] = -beta_hat*p[k];
                    M_hat[(k+1)*(2*i+2)-1] = static_cast<realdp> (0);
                    M_tilde[(k+1)*(2*i+2)-2] = -beta_tilde*p_tilde[k];
                    M_tilde[(k+1)*(2*i+2)-1] = static_cast<realdp> (0);
                }
                for (integer j = 0; j < 2*i; j++)
                {
                    M_hat[(2*i)*(2*i+2)+j] = -beta_hat*p[j];
                    M_hat[(2*i+1)*(2*i+2)+j] = static_cast<realdp> (0);
                    M_tilde[(2*i)*(2*i+2)+j] = -beta_tilde*p_tilde[j];
                    M_tilde[(2*i+1)*(2*i+2)+j] = static_cast<realdp> (0);
                }
                M_hat[(2*i+1)*(2*i+2)-2] = -beta_hat;
                M_hat[(2*i+1)*(2*i+2)-1] = static_cast<realdp> (0);
                M_hat[(2*i+2)*(2*i+2)-2] = static_cast<realdp> (0);
                M_hat[(2*i+2)*(2*i+2)-1] = static_cast<realdp> (0);
                M_tilde[(2*i+1)*(2*i+2)-2] = -beta_tilde;
                M_tilde[(2*i+1)*(2*i+2)-1] = static_cast<realdp> (0);
                M_tilde[(2*i+2)*(2*i+2)-2] = static_cast<realdp> (0);
                M_tilde[(2*i+2)*(2*i+2)-1] = static_cast<realdp> (0);
            }
        }
//        std::cout << "tttest:" << M_tilde[0] << std::endl;//---
        for (integer i = 0; i < Currentlength; i++)
        {
            idx = (beginidx + i) % LengthSY;
            psiv[2*i] = Mani->Metric(x1,S[idx],v);
            psiv[2*i+1] = Mani->Metric(x1,Y[idx],v)*gamma;
            if (ifSR1[i] == 1)
            {
                psiv[2*i] = psiv[2*i+1] - psiv[2*i];
                psiv[2*i+1] = static_cast<realdp> (0);
            }
        }
        for (integer i = 0; i < 2*Currentlength; i++)
        {
            Mpsiv[i] = static_cast<realdp> (0);
//            std::cout << "test:" << Mpsiv[i] << std::endl;//---
            for (integer j = 0; j < 2*Currentlength; j++)
            {
                Mpsiv[i] = Mpsiv[i] + M_tilde[j+i*(2*Currentlength)]*psiv[j];
//                std::cout << i << ":" << Mpsiv[i] << ", :" << M_tilde[j+i*(2*Currentlength)] << ", :" << psiv[j] << std::endl;//---
            }
        }
        
//        result->Print("r1:");
        Prob->PreConditioner(x1, *result, &Py);
//        result->Print("r2:");
        Mani->ScalarTimesVector(x1, gamma, Py, result);
//        result->Print("r3:");
        for (integer i = 0; i < Currentlength; i++)
        {
            idx = (beginidx + i) % LengthSY;
            if (ifSR1[i] == 0)
            {
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i], S[idx], *result, result);
//                result->Print("r4:");
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i+1]*gamma, Y[idx], *result, result);
//                result->Print("r5:");
            }
            else
            {
//                std::cout << "ttt:" << -Mpsiv[2*i] << std::endl;//---
//                S[idx].Print("here:");//---
                Mani->ScalarVectorAddVector(x1, -Mpsiv[2*i], S[idx], *result, result);
//                result->Print("r6:");
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i]*gamma, Y[idx], *result, result);
//                result->Print("r7:");
            }
        }

        delete[] M_hat;
        delete[] M_tilde;
        delete[] psis;
        delete[] psiy;
        delete[] p;
        delete[] p_tilde;
        delete[] psiv;
        delete[] Mpsiv;
        delete[] ifSR1;

        return *result;
    };
    
    realdp LRBroydenFamily::InitialHessian(realdp inpss, realdp inpsy, realdp inpyy)
    { /*Suggested in NW2006*/
        return inpsy / inpyy;
    };
    
    void LRBroydenFamily::UpdateDataLRBroydenFamily(void)
    {
        Mani->VectorTransport(x1, eta2, x2, eta2, &s); nV++;
        Vector Tgf1(gf1); Mani->VectorTransport(x1, eta2, x2, gf1, &Tgf1); nVp++;
        betay = Mani->Beta(x1, eta2);
        Mani->VectorLinearCombination(x2, static_cast<realdp> (1) / betay, gf2, -1, Tgf1, &y);
        Prob->PreConditioner(x2, y, &Py);
        
        inpsy = Mani->Metric(x2, s, y);
        inpss = Mani->Metric(x2, s, s);
        inpyy = Mani->Metric(x2, y, Py);
        if (inpsy / inpss >= nu * pow(ngf2, mu) && (ngf2 / ngf0 < 1e-3 ||
                                                    (inpss > std::numeric_limits<realdp>::epsilon() && inpsy > std::numeric_limits<realdp>::epsilon())))
        {
            gamma = InitialHessian(inpss, inpsy, inpyy);
//            gamma_ave += gamma;
            if(LMrestart && Currentlength >= LengthSY)
                Currentlength = 0;
            
            if (Currentlength < LengthSY)
            {
                Y[Currentlength] = y;
                S[Currentlength] = s;
                store_phi[Currentlength] = phi;
                store_sr[Currentlength] = sr_value;
                SY[Currentlength*LengthSY + Currentlength] = inpsy;
                SS[Currentlength*LengthSY + Currentlength] = inpss;
                YY[Currentlength*LengthSY + Currentlength] = inpyy;
                for (integer i = 0; i < Currentlength; i++)
                {
                    Mani->VectorTransport(x1, eta2, x2, Y[i], &Y[i]); nVp++;
                    Mani->VectorTransport(x1, eta2, x2, S[i], &S[i]); nVp++;
                }
                for (integer i = 0; i < Currentlength; i++)
                {
                    SY[i*LengthSY + Currentlength] = Mani->Metric(x2, S[i], Y[Currentlength]);
                    SY[Currentlength*LengthSY + i] = Mani->Metric(x2, S[Currentlength], Y[i]);
                    SS[i*LengthSY + Currentlength] = Mani->Metric(x2, S[i], S[Currentlength]);
                    SS[Currentlength*LengthSY + i] = SS[i*LengthSY + Currentlength];
                    YY[i*LengthSY + Currentlength] = Mani->Metric(x2, Y[i], Y[Currentlength]);
                    YY[Currentlength*LengthSY + i] = YY[i*LengthSY + Currentlength];
                }
                Currentlength++;
            }
            else
                if (LengthSY > 0)
                {
                    integer idx;
                    Y[beginidx] = y;
                    S[beginidx] = s;
                    store_phi[beginidx] = phi;
                    store_sr[beginidx] = sr_value;
                    SY[beginidx*LengthSY + beginidx] = inpsy;
                    SS[beginidx*LengthSY + beginidx] = inpss;
                    YY[beginidx*LengthSY + beginidx] = inpyy;
                    tempBegin = beginidx;  //store the beginidx for the last loop
                    beginidx = (++beginidx) % LengthSY;
                    for (integer i = beginidx; i < beginidx + LengthSY - 1; i++)
                    {
                        idx = i % LengthSY;
                        Mani->VectorTransport(x1, eta2, x2, Y[idx], &Y[idx]); nVp++;
                        Mani->VectorTransport(x1, eta2, x2, S[idx], &S[idx]); nVp++;
                    }
                    for (integer i = beginidx; i < beginidx + LengthSY - 1; i++)
                    {
                        idx = i % LengthSY;
                        SY[tempBegin * LengthSY + idx] = Mani->Metric(x2, S[tempBegin], Y[idx]);
                        SY[LengthSY * idx + tempBegin] = Mani->Metric(x2, S[idx], Y[tempBegin]);
                        SS[tempBegin * LengthSY + idx] = Mani->Metric(x2, S[tempBegin], S[idx]);
                        SS[LengthSY * idx + tempBegin] = SS[tempBegin * LengthSY + idx];
                        YY[tempBegin * LengthSY + idx] = Mani->Metric(x2, Y[tempBegin], Y[idx]);
                        YY[LengthSY * idx + tempBegin] = YY[tempBegin * LengthSY + idx];
                    }
                }
            isupdated = true;
        }
        else
        {
            for (integer i = 0; i < Currentlength; i++)
            {
                Mani->VectorTransport(x1, eta2, x2, Y[i], &Y[i]); nVp++;
                Mani->VectorTransport(x1, eta2, x2, S[i], &S[i]); nVp++;
            }
            isupdated = false;
        }
    };
    
}; /*end of ROPTLIB namespace*/
