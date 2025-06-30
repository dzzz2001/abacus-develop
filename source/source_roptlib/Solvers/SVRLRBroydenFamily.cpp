
#include "Solvers/SVRLRBroydenFamily.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    SVRLRBroydenFamily::SVRLRBroydenFamily(const Problem *prob, const Variable *initialx)
    {
        Initialization(prob, initialx);
    };
    
    void SVRLRBroydenFamily::SetProbX(const Problem *prob, const Variable *initialx)
    {
        SolversSMSVRG::SetProbX(prob, initialx);
        prob->SetUseGrad(true);
        prob->SetUseHess(true);
        s = Prob->GetDomain()->GetEMPTY();
        y = Prob->GetDomain()->GetEMPTY();
        Py = Prob->GetDomain()->GetEMPTY();
    };
    
    void SVRLRBroydenFamily::SetDefaultParams(void)
    {
        SolversSMSVRG::SetDefaultParams();
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
        //RHO = nullptr;
        M_tilde = nullptr;
        ifSR1 = nullptr;
        Currentlength = 0;
        beginidx = 0;
        gamma = 1;
        SolverName.assign("SVRLRBroydenFamily");
    };
    
    void SVRLRBroydenFamily::SetParams(PARAMSMAP params)
    {
        SolversSMSVRG::SetParams(params);
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
    
    SVRLRBroydenFamily::~SVRLRBroydenFamily(void)
    {
        DeleteVectors(S, LengthSY);
        DeleteVectors(Y, LengthSY);
        //if (RHO != nullptr)
         //   delete[] RHO;
        if (SY != nullptr)
            delete[] SY;
        if (SS != nullptr)
            delete[] SS;
        if (YY != nullptr)
            delete[] YY;
        if (M_tilde != nullptr)
            delete[] M_tilde;
        if (ifSR1 != nullptr)
            delete[] ifSR1;
    };
    
    void SVRLRBroydenFamily::Run(void)
    {
        DeleteVectors(S, LengthSY);
        NewVectors(S, LengthSY);
        DeleteVectors(Y, LengthSY);
        NewVectors(Y, LengthSY);
        //if (RHO != nullptr)
          //  delete[] RHO;
        if (M_tilde != nullptr)
            delete[] M_tilde;
        if (ifSR1 != nullptr)
            delete[] ifSR1;
        if (SY != nullptr)
            delete[] SY;
        if (SS != nullptr)
            delete[] SS;
        if (YY != nullptr)
            delete[] YY;
        //RHO = new realdp[LengthSY];
        M_tilde = new realdp[4*LengthSY*LengthSY];
        ifSR1 = new integer[LengthSY];
        SY = new realdp[LengthSY*LengthSY];
        SS = new realdp[LengthSY*LengthSY];
        YY = new realdp[LengthSY*LengthSY];
        SolversSMSVRG::Run();
    };
    
    void SVRLRBroydenFamily::CheckParams(void)
    {
        SolversSMSVRG::CheckParams();
        char YES[] = "YES";
        char NO[] = "NO";
        char *status;
        
        printf("LR-SBroyden-VR METHOD PARAMETERS:\n");
        status = (nu >= 0 && nu < 1) ? YES : NO;
        printf("nu            :%15g[%s],\t", nu, status);
        status = (mu >= 0) ? YES : NO;
        printf("mu            :%15g[%s],\n", mu, status);
        status = YES;
        printf("isconvex      :%15d[%s],\t", isconvex, status);
        status = (LengthSY >= 0) ? YES : NO;
        printf("LengthSY      :%15d[%s],\n", LengthSY, status);
        status = YES;
        printf("LMrestart     :%15d[%s],\n", LMrestart, status);
    };
    
    void SVRLRBroydenFamily::GetSearchDir(void)
    {
        HvSVRLRBroydenFamily(gf1, &eta1);
        Mani->ScalarTimesVector(x1, -1.0, eta1, &eta1);
    };
    
    void SVRLRBroydenFamily::UpdateData(void)
    {
        UpdateDataSVRLRBroydenFamily();
    };
    
    void SVRLRBroydenFamily::PrintInfo(void)
    {
        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));
        
        printf("|gf|:%.3e,t0:%.2e,t:%.2e,time:%.2g,", ngf2, initiallength, stepsize, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        
        printf("\n\tbetay:%.3e,gamma:%.3e,inpss:%.3e,inpsy:%.3e,inpyy:%.3e,IsUpdateHessian:%d,", betay, gamma, inpss, inpsy, inpyy, isupdated);
        
        printf("nf:%d,ng:%d,nsf:%d,nsg:%d,", nf, ng, nsf, nsg);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
    };
    
    void SVRLRBroydenFamily::HessianPrep(void)
    {
        if (Currentlength > 0)
        {
        integer idx;
        
        realdp sBs,yHy,yTs,phi,phic,Bphi;
        realdp alpha_hat,alpha_tilde,beta_hat,beta_tilde,delta_hat,delta_tilde;
        realdp ppsis,ppsiy;
        realdp *M_hat = new realdp[4*Currentlength*Currentlength];
        realdp *psis = new realdp[2*Currentlength]; //Psi'*s
        realdp *psiy = new realdp[2*Currentlength]; //Psi_tilde'*y
        realdp *p = new realdp[2*Currentlength]; //M_hat*(Psi'*s)
        realdp *p_tilde = new realdp[2*Currentlength]; //M_tilde*(Psi_tilde'*y)
        realdp apb, bcube, kbfgs, koc; //condition number for hybrid Davidon
        /* initial  */
        sBs = SS[beginidx*LengthSY + beginidx]/gamma;
        yHy = YY[beginidx*LengthSY + beginidx]*gamma;
        yTs = SY[beginidx*LengthSY + beginidx];
        phic = (yTs*yTs)/(yTs*yTs-sBs*yHy);
        //-----------------------------------------------
        apb = yHy + yTs;
        bcube = yTs * yTs * yTs;
        kbfgs = (apb*apb*sBs - 2*bcube + apb*sqrt(apb*apb*sBs*sBs-4*bcube*sBs))/(2*bcube);
        if (yTs <= static_cast<realdp> (2) *sBs*yHy/(sBs+yHy))
        {
            phi = yTs*(yHy-yTs)/(sBs*yHy-yTs*yTs);
            ifSR1[0] = 0;
            koc = (2*yHy*yTs - yTs*yTs + 2*sqrt(yHy*yHy*sBs*sBs-yHy*yTs*yTs*sBs))/(yTs*yTs);
        }
        else
        {
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
        if (kbfgs < phiset * koc)
        {
            phi = 0;
            ifSR1[0] = 0;
        }
        //------------------------------------------------------------
        //std::cout << phi << std::endl; //output phi
        //Bphi = (1-phi)/(1-phi+RHO[idx]*RHO[idx]*phi*sBs*yHy);
        Bphi = (1-phi)/(1-phi+phi*sBs*yHy/(yTs*yTs));
        if (ifSR1[0] == 1)
        {
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
            M_hat[0] = -beta_hat;
            M_hat[1] = 0.;
            M_hat[2] = 0.;
            M_hat[3] = 0.;
            M_tilde[0] = -beta_tilde;
            M_tilde[1] = 0.;
            M_tilde[2] = 0.;
            M_tilde[3] = 0.;
        }
        else
        {
            alpha_hat = (phi-1) / sBs;
            alpha_tilde = (1 + Bphi * yHy / yTs) / yTs;
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
            delta_hat = (1 + phi * sBs / yTs) / yTs;
            delta_tilde = (Bphi-1) / yHy;
            M_hat[0] = alpha_hat;
            M_hat[1] = beta_hat;
            M_hat[2] = beta_hat;
            M_hat[3] = delta_hat;
            M_tilde[0] = alpha_tilde;
            M_tilde[1] = beta_tilde;
            M_tilde[2] = beta_tilde;
            M_tilde[3] = delta_tilde;
        }
        
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
                    psis[2*j+1] = 0.;
                    psiy[2*j] = psiy[2*j+1] - psiy[2*j];
                    psiy[2*j+1] = 0.;
                }
            }
            for (integer j = 0; j < 2*i; j++)
            {
                p[j] = 0.;
                p_tilde[j] = 0.;
                for (integer k = 0; k < 2*i; k++)
                {
                    p[j] = p[j] + M_hat[j*(2*i)+k] * psis[k];
                    p_tilde[j] = p_tilde[j] + M_tilde[j*(2*i)+k] * psiy[k];
                }
            }
            ppsis = 0.;
            ppsiy = 0.;
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
            if (kbfgs < phiset * koc)
            {
                phi = 0;
                ifSR1[i] = 0;
            }
            //------------------------------------------------------------
            Bphi = (1-phi)/(1-phi+phi*sBs*yHy/(yTs*yTs));
            beta_hat = -phi / yTs;
            beta_tilde =  -Bphi / yTs;
            if (ifSR1[i] == 0)
            {
                alpha_hat = (phi-1) / sBs;
                alpha_tilde = (static_cast<realdp> (1) + Bphi * yHy / yTs) / yTs;
                delta_hat = (static_cast<realdp> (1) + phi * sBs / yTs) / yTs;
                delta_tilde = (Bphi-1) / yHy;
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
                    M_hat[(k+1)*(2*i+2)-1] = 0.;
                    M_tilde[(k+1)*(2*i+2)-2] = -beta_tilde*p_tilde[k];
                    M_tilde[(k+1)*(2*i+2)-1] = 0.;
                }
                for (integer j = 0; j < 2*i; j++)
                {
                    M_hat[(2*i)*(2*i+2)+j] = -beta_hat*p[j];
                    M_hat[(2*i+1)*(2*i+2)+j] = 0.;
                    M_tilde[(2*i)*(2*i+2)+j] = -beta_tilde*p_tilde[j];
                    M_tilde[(2*i+1)*(2*i+2)+j] = 0.;
                }
                M_hat[(2*i+1)*(2*i+2)-2] = -beta_hat;
                M_hat[(2*i+1)*(2*i+2)-1] = 0.;
                M_hat[(2*i+2)*(2*i+2)-2] = 0.;
                M_hat[(2*i+2)*(2*i+2)-1] = 0.;
                M_tilde[(2*i+1)*(2*i+2)-2] = -beta_tilde;
                M_tilde[(2*i+1)*(2*i+2)-1] = 0.;
                M_tilde[(2*i+2)*(2*i+2)-2] = 0.;
                M_tilde[(2*i+2)*(2*i+2)-1] = 0.;
            }
        }
        
        delete[] M_hat;
        delete[] psis;
        delete[] psiy;
        delete[] p;
        delete[] p_tilde;
        }
    };
    
    Vector &SVRLRBroydenFamily::HvSVRLRBroydenFamily(const Vector &v, Vector *result)
    {
        integer idx;
        *result = v;
        
        if (Currentlength == 0)
        {
            Prob->PreConditioner(x1, *result, &Py);
            Mani->ScalarTimesVector(x1, gamma, Py, result);
            return *result;
        }
        
        realdp *psiv = new realdp[2*Currentlength]; //Psi_tilde'*v
        realdp *Mpsiv = new realdp[2*Currentlength]; //M_tilde*(Psi_tilde'*v)
        
        for (integer i = 0; i < Currentlength; i++)
        {
            idx = (beginidx + i) % LengthSY;
            psiv[2*i] = Mani->Metric(x1,S[idx],v);
            psiv[2*i+1] = Mani->Metric(x1,Y[idx],v)*gamma;
            if (ifSR1[i] == 1)
            {
                psiv[2*i] = psiv[2*i+1] - psiv[2*i];
                psiv[2*i+1] = 0;
            }
        }
        for (integer i = 0; i < 2*Currentlength; i++)
        {
            Mpsiv[i] = 0.;
            for (integer j = 0; j < 2*Currentlength; j++)
            {
                Mpsiv[i] = Mpsiv[i] + M_tilde[j+i*(2*Currentlength)]*psiv[j];
            }
        }
        
        Prob->PreConditioner(x1, *result, &Py);
        Mani->ScalarTimesVector(x1, gamma, Py, result);
        for (integer i = 0; i < Currentlength; i++)
        {
            idx = (beginidx + i) % LengthSY;
            if (ifSR1[i] == 0)
            {
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i], S[idx], *result, result);
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i+1]*gamma, Y[idx], *result, result);
            }
            else
            {
                Mani->ScalarVectorAddVector(x1, -Mpsiv[2*i], S[idx], *result, result);
                Mani->ScalarVectorAddVector(x1, Mpsiv[2*i]*gamma, Y[idx], *result, result);
            }
        }
        
        delete[] psiv;
        delete[] Mpsiv;

        return *result;
    }
    
    realdp SVRLRBroydenFamily::InitialHessian(realdp inpss, realdp inpsy, realdp inpyy)
    { /*Suggested in NW2006*/
        return inpsy / inpyy;
    };
    
    void SVRLRBroydenFamily::UpdateDataSVRLRBroydenFamily(void)
    {
        Vector eta3(eta2);
//        eta3.Print("eta31:");//---
        Mani->InvRetraction(x1, x2, &eta3);
//        eta3.Print("eta32:");//---
        Mani->VectorTransport(x1, eta3, x2, eta3, &s); nV++;
//        eta3.Print("eta33:");//---
//        gf1.Print("gf1:");//---
        Vector Tgf1(gf1); Mani->VectorTransport(x1, eta3, x2, gf1, &Tgf1); nVp++;
        betay = Mani->Beta(x1, eta3);
        Mani->VectorLinearCombination(x2, static_cast<realdp> (1) / betay, gf2, -1, Tgf1, &y);
        Prob->PreConditioner(x2, y, &Py);
        
        inpsy = Mani->Metric(x2, s, y);
        inpss = Mani->Metric(x2, s, s);
        inpyy = Mani->Metric(x2, y, Py);
        //rho = static_cast<realdp> (1) / inpsy;
        if (inpsy / inpss >= nu * pow(ngf2, mu) && (ngf2 / ngf0 < 1e-3 ||
                                                    (inpss > std::numeric_limits<realdp>::epsilon() && inpsy > std::numeric_limits<realdp>::epsilon())))
        {
            gamma = InitialHessian(inpss, inpsy, inpyy);
            //------------------------------------------------------------
            //std::cout<< gamma << std::endl;
            if(LMrestart && Currentlength >= LengthSY)
                Currentlength = 0;
            
            if (Currentlength < LengthSY)
            {
                Y[Currentlength] = y;
                S[Currentlength] = s;
                //RHO[Currentlength] = rho;
                SY[Currentlength*LengthSY + Currentlength] = inpsy;
                SS[Currentlength*LengthSY + Currentlength] = inpss;
                YY[Currentlength*LengthSY + Currentlength] = inpyy;
                for (integer i = 0; i < Currentlength; i++)
                {
                    Mani->VectorTransport(x1, eta3, x2, Y[i], &Y[i]); nVp++;
                    Mani->VectorTransport(x1, eta3, x2, S[i], &S[i]); nVp++;
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
                    //RHO[beginidx] = rho;
                    SY[beginidx*LengthSY + beginidx] = inpsy;
                    SS[beginidx*LengthSY + beginidx] = inpss;
                    YY[beginidx*LengthSY + beginidx] = inpyy;
                    tempBegin = beginidx;  //store the beginidx for the last loop
                    beginidx = (++beginidx) % LengthSY;
                    for (integer i = beginidx; i < beginidx + LengthSY - 1; i++)
                    {
                        idx = i % LengthSY;
                        Mani->VectorTransport(x1, eta3, x2, Y[idx], &Y[idx]); nVp++;
                        Mani->VectorTransport(x1, eta3, x2, S[idx], &S[idx]); nVp++;
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
                Mani->VectorTransport(x1, eta3, x2, Y[i], &Y[i]); nVp++;
                Mani->VectorTransport(x1, eta3, x2, S[i], &S[i]); nVp++;
            }
            isupdated = false;
        }
    };
    
}; /*end of ROPTLIB namespace*/
