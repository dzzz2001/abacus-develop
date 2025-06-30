
#include "Solvers/SolversSMSVRG.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    void SolversSMSVRG::Run(void)
    {
        Variable xTemp(x1);
        Vector gfTemp = Prob->GetDomain()->GetEMPTY();
        Vector gfTemp1 = Prob->GetDomain()->GetEMPTY();
        Vector gfTemp2 = Prob->GetDomain()->GetEMPTY();
        Vector gfTempBatch = Prob->GetDomain()->GetEMPTY();
        Vector invReEta = Prob->GetDomain()->GetEMPTY();
        Vector gfoutput = Prob->GetDomain()->GetEMPTY();
        Vector BatchIndex(BatchSize);
        BatchIndex.NewMemoryOnWrite();
//        integer it[BatchSize];
        integer ifHHR = 0; // sweep = 0,
        
        SolversSM::Run();
        
        f1 = Prob->f(x1); nf++;
        f2 = f1;
        Prob->Grad(x1, &gf1); ng++;
        ngf0 = sqrt(Mani->Metric(x1, gf1, gf1));
        ngf1 = ngf0; ngf2 = ngf1;
        iter = 0;
        ChooseStepsize();
        (this->*StepsizePtr)();
//        std::cout << "stepsize0:" << stepsize << std::endl;//---
//        stepsize = 0;
        Vector initialstepsize(1 + Max_Iteration);
        realdp *timeSeriesptr = timeSeries.ObtainWritePartialData();
        realdp *funSeriesptr = funSeries.ObtainWritePartialData();
        realdp *gradSeriesptr = gradSeries.ObtainWritePartialData();
        realdp *initialstepsizeptr = initialstepsize.ObtainWritePartialData();
        
        if (Verbose >= FINALRESULT)
            printf("i:%d,f:%.3e,|gf|:%.3e,\n", iter, f1, ngf1);
        if (Verbose >= ITERRESULT)
        {
            timeSeriesptr[iter] = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
            funSeriesptr[iter] = f1;
            gradSeriesptr[iter] = ngf1;
            initialstepsizeptr[iter] = 0;
        }
        bool isstop = IsStopped();
        /*Start the loop*/
        while ((((! isstop) && iter < Max_Iteration) || iter < Min_Iteration))
        {
            xTemp = x1;
            gfTemp = gf1; /* full gradient */
            HessianPrep(); //Generate \tilde{M} and ifSR for Hessian approximation in LS-SBroyden
            for (integer i = 0; i < FreqM; i++)
            {   
                if (i == 0)
                {
                    GetSearchDir();
                    (this->*StepsizePtr)();
//                    std::cout << "stepsize1:" << stepsize << std::endl;//---
                    Mani->ScalarTimesVector(xTemp, stepsize, eta1, &eta2);
                    Mani->Retraction(xTemp, eta2, &x2);
                    xTemp = x2; //update xTemp
                }
                else
                {
                    for (integer j = 0; j < BatchSize; j++)
                    {
                        BatchIndex[j] = static_cast<integer> ((Prob->GetDataSize()) * genrandreal());
                    }
//                    BatchIndex.Print("batchindex:");//----
                    
                    Prob->StoGrad(x1, &gfTemp1, BatchIndex); nsg++; //gfTemp1 = grad f(w_k) it-th
                    Prob->Stof(xTemp, BatchIndex);
                    Prob->StoGrad(xTemp, &gfTemp2, BatchIndex); nsg++; //gfTemp2 = grad f(x_t) it-th
                    
                    Mani->VectorLinearCombination(x1, 1, gfTemp, -1, gfTemp1, &gfTemp1);
                    
                    if (Prob->GetDomain()->GetHasHHR())
                    {
                        Prob->GetDomain()->SetHasHHR(false);
                        ifHHR = 1;
                    } //temporary remove the need for locking condition
                    Mani->InvRetraction(x1, xTemp, &invReEta);
                    Mani->InverseVectorTransport(x1, invReEta, xTemp, gfTemp2, &gfTemp2); //xTemp.field problem
                    
                    Mani->VectorLinearCombination(x1, 1, gfTemp1, 1, gfTemp2, &gf1);
                    
                    GetSearchDir();
                    
                    Mani->VectorTransport(x1, invReEta, xTemp, eta1, &eta1);
                    if (ifHHR == 1)
                    {
                        Prob->GetDomain()->SetHasHHR(true);
                        ifHHR = 0;
                    }
                    
                    (this->*StepsizePtr)();
//                    std::cout << "stepsize2:" << stepsize << std::endl;//---
                    Mani->ScalarTimesVector(xTemp, stepsize, eta1, &eta2);
                    Mani->Retraction(xTemp, eta2, &x2);
                    xTemp = x2; //update xTemp
                }
            }
            
            initiallength = stepsize;
            if (Verbose >= ITERRESULT)
                initialstepsizeptr[iter + 1] = initiallength;
            
            f2 = Prob->f(x2); nf++;
            Prob->Grad(x2, &gf2); ng++;  //update f2,gf2
            /*norm of the gradient at x2*/
            iter++; // full graident computation recorded
            
            if (Verbose >= ITERRESULT)
            {
                /*Output information*/
                if (iter % OutputGap == 0)
                    PrintInfo(); // Output information specific to Algorithms
                
                /*Store debug information in the arrays*/
                timeSeriesptr[iter] = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
                funSeriesptr[iter] = f2; gradSeriesptr[iter] = sqrt(Mani->Metric(x2, gf2, gf2));
            }
            
            if (std::isnan(f2) || std::isinf(f2)) /*Stop when got a nan or inf*/
            {
                printf("New function value is either nan or inf. Stop!\n");
                break;
            }
            
            gf1 = gfTemp;   //for UpdateData()
//            eta2.Print("eta2:");//---
//            gf1.Print("gf1:");//---
            UpdateData(); //Update the Hessian approximation for quasi-Newton methods
            
            ngf2 = sqrt(Mani->Metric(x2, gf2, gf2));
            /*Call the function to check whether the stopping criterion is satisfied or not.
             The default function is written in Solvers.h and Solvers.cpp*/
            isstop = IsStopped();
            
            //            xTemp.Delete();
            xTemp = x1; x1 = x2; x2 = xTemp;
            f1 = f2;
            gfTemp = gf1; gf1 = gf2; gf2 = gfTemp;
            ngf1 = ngf2;
        }
        
        ComTime = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
        {
            lengthSeries = iter + 1;
            SolverInfo.AddToFields("initialstepsize", initialstepsize);
        }
        if (Verbose >= FINALRESULT)
            PrintFinalInfo();
    };
    
//    void SolversSMSVRG::InitialStepSize(void)
//    {
//        stepsize = Initstepsize;
//    }
//    
//    realdp SolversSMSVRG::h(void)
//    {
//        Mani->ScalarTimesVector(x1, stepsize, eta1, &eta2);
//        Mani->Retraction(x1, eta2, &x2); nR++;
//        nf++;
//        return Prob->f(x2);
//    };
//
//    realdp SolversSMSVRG::dh(void)
//    {
//        //        std::cout << "dh 1" << std::endl;//---
//        Prob->Grad(x2, &gf2); ng++;
//        //        std::cout << "dh 2" << std::endl;//---
//        nV++;
//        Vector diffeta1(Mani->GetEMPTY());
//        Mani->VecTranDiffRet(x1, eta2, x2, eta1, &diffeta1, true);
//        realdp tmp = Mani->Metric(x2, gf2, diffeta1);
//        //        std::cout << "dh 3" << std::endl;//---
//        return tmp;
//    };

    void SolversSMSVRG::CheckParams(void)
    {
        SolversSMSto::CheckParams();
        
        char YES[] = "YES";
        char NO[] = "NO";
        char *status;
        
        printf("STOCHASTIC VARIANCE REDUCTION TYPE METHODS PARAMETERS:\n");
//        status = (Initstepsize > 0) ? YES : NO;
//        printf("Initstepsize  :%15g[%s],\t", Initstepsize, status);
//        status = (Decaying_rate > 0 && Decaying_rate < 1) ? YES : NO;
//        printf("Decaying_rate  :%15g[%s],\n", Decaying_rate, status);
//        status = (Initstepsize > 0) ? YES : NO;
//        printf("Initstepsize   :%15g[%s],\t", Initstepsize, status);
//        status = YES;
//        printf("Finalstepsize :%15g[%s],\n", Finalstepsize, status);
//        status = (Minstepsize > 0 && Minstepsize <= Maxstepsize) ? YES : NO;
//        printf("Minstepsize   :%15g[%s],\t", Minstepsize, status);
//        status = (Maxstepsize > 0 && Maxstepsize >= Minstepsize) ? YES : NO;
//        printf("Maxstepsize   :%15g[%s],\n", Maxstepsize, status);
        status = (FreqM > 0) ? YES : NO;
        printf("FreqM          :%15d[%s],\n", FreqM, status);
//        status = (BatchSize > 0) ? YES : NO;
//        printf("BatchSize      :%15d[%s],\n", BatchSize, status);
    };
    
    void SolversSMSVRG::UpdateData(void)
    {
    };
    
    void SolversSMSVRG::HessianPrep(void)
    {
    };
    
//    void SolversSMSVRG::PrintInfo(void)
//    {
//        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));
//        
//        printf("|gf|:%.3e,t0:%.2e,t:%.2e,time:%.2g,", ngf2, initiallength, stepsize, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        
//        printf("nf:%d,ng:%d,nsf:%d,nsg:%d,", nf, ng, nsf, nsg);
//        
//        if (nH != 0)
//            printf("nH:%d,", nH);
//        
//        printf("nR:%d,", nR);
//        
//        if (nV != 0)
//            printf("nV(nVp):%d(%d),", nV, nVp);
//        
//        printf("\n");
//    };
//    
//    void SolversSMSVRG::PrintFinalInfo(void)
//    {
//        printf("i:%d,f:%.3e,", iter, f2);
//        
//        printf("|gf|:%.3e,|gf|/|gf0|:%.3e,time:%.2g,", ngf2, ngf2/ngf0, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        
//        printf("nf:%d,ng:%d,nsf:%d,nsg:%d,", nf, ng, nsf, nsg);
//        
//        if (nH != 0)
//            printf("nH:%d,", nH);
//        
//        printf("nR:%d,", nR);
//        
//        if (nV != 0)
//            printf("nV(nVp):%d(%d),", nV, nVp);
//        
//        printf("\n");
//    };
    
    void SolversSMSVRG::SetDefaultParams()
    {
        SolversSMSto::SetDefaultParams();
        Initstepsize = static_cast<realdp> (1e-2);
//        Decaying_rate = static_cast<realdp> (0.999);
//        Minstepsize = std::numeric_limits<realdp>::epsilon();
//        Maxstepsize = static_cast<realdp> (1e10);
//        Initstepsize = 1;
//        Finalstepsize = -1;
        FreqM = 10;
//        BatchSize = 1;
        nsf = 0;
        nsg = 0;
    };
    
    SolversSMSVRG::~SolversSMSVRG(void)
    {
    };
    
    void SolversSMSVRG::SetParams(PARAMSMAP params)
    {
        SolversSMSto::SetParams(params);
        PARAMSMAP::iterator iter;
        for (iter = params.begin(); iter != params.end(); iter++)
        {
//            if (iter->first == static_cast<std::string> ("Initstepsize"))
//            {
//                Initstepsize = iter->second;
//            }
//            else
//            if (iter->first == static_cast<std::string> ("Decaying_rate"))
//            {
//                Decaying_rate = iter->second;
//            }
//            else
//            if (iter->first == static_cast<std::string> ("Minstepsize"))
//            {
//                Minstepsize = iter->second;
//            }
//            else
//            if (iter->first == static_cast<std::string> ("Maxstepsize"))
//            {
//                Maxstepsize = iter->second;
//            }
//            else
//            if (iter->first == static_cast<std::string> ("Initstepsize"))
//            {
//                Initstepsize = iter->second;
//            }
//            else
//            if (iter->first == static_cast<std::string> ("Finalstepsize"))
//            {
//                Finalstepsize = iter->second;
//            }
//            else
            if (iter->first == static_cast<std::string> ("FreqM"))
            {
                FreqM = iter->second;
            }
//            else
//            if (iter->first == static_cast<std::string> ("BatchSize"))
//            {
//                BatchSize = iter->second;
//            }
        }
    };
}; /*end of ROPTLIB namespace*/
