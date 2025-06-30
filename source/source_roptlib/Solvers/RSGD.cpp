#include "Solvers/RSGD.h"

/*Define the namespace*/
namespace ROPTLIB {
	RSGD::RSGD(const Problem *prob, const Variable *initialx)
	{
		Initialization(prob, initialx);
	};

	void RSGD::Run(void)
	{
		
		Variable xTemp(x1);
		Vector gfTemp = Prob->GetDomain()->GetEMPTY();
		Vector gradSum = Prob->GetDomain()->GetEMPTY();
		gradSum.SetToZeros();

		SolversSMSto::Run();
        integer N = Prob->GetDataSize(); // num
        integer numBatch = static_cast<integer>(N / BatchSize); // Batch
        Vector indexall(N);
        indexall.NewMemoryOnWrite();
        for(integer i = 0; i < N; i++)
            indexall[i] = i;
        
        
//		integer batchsize = Prob->BatchSize;
//		integer num = Prob->num; //number of sample
//		integer Batch = Prob->GetBatch();
//		integer *ind = new integer[num];
//		Prob->GenerateInitialIndex(num, ind);
//
//		Prob->BatchIndex = ind;
//		Prob->BatchSize = num;
//		f1 = Prob->f(x1); nf++;
//		f2 = f1;
//		Prob->Grad(x1, &gradSum); ng++;
//		gf1 = gradSum;
//		Prob->BatchSize = batchsize;
        f1 = Prob->f(x1); nf++;
        f2 = f1;
        Prob->Grad(x1, &gradSum); ng++;
        gf1 = gradSum;
		
		ngf0 = sqrt(Mani->Metric(x1, gf1, gf1));
		ngf1 = ngf0; ngf2 = ngf1;
		iter = 0;
//		Prob->burnin = true;
		ChooseStepsize();
		(this->*StepsizePtr)();
		
		realdp *timeSeriesptr = timeSeries.ObtainWritePartialData();
		realdp *funSeriesptr = funSeries.ObtainWritePartialData();
		realdp *gradSeriesptr = gradSeries.ObtainWritePartialData();
//		xSeries.NewMemoryOnWrite();

		if (Verbose >= FINALRESULT)
			printf("i:%d,f:%.3e,|gf|:%.3e,\n", iter, f1, ngf1);
		
		if (Verbose >= ITERRESULT)
		{
			timeSeriesptr[iter] = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
			funSeriesptr[iter] = f1;
			gradSeriesptr[iter] = ngf1;
//			xSeries.GetElement(iter) = x1;
		}
		bool isstop = IsStopped();

		/*Start the loop*/
		while (((!isstop) && iter < Max_Iteration) || iter < Min_Iteration)
		{
//			Prob->burnin = iter <= FixedStep ? true : false;
			if (numBatch == 1) /* if only one batch, then use full gradient descent */
			{
				Mani->ScalarTimesVector(x1, -stepsize, gf1, &eta2);
				Mani->Retraction(x1, eta2, &x2); nR++;
				f2 = Prob->f(x2); nf++;
				Prob->Grad(x2, &gf2); ng++;

				if (std::isnan(f2) || std::isinf(f2)) /*Stop when got a nan or inf*/
				{
					printf("New function value is either nan or inf. Stop!\n");
					break;
				}

				/*Switch information at x1 and x2*/
				xTemp = x1;	x1 = x2; x2 = xTemp;
				gfTemp = gf1; gf1 = gf2; gf2 = gfTemp;

				gradSum = gf1;
			}
			else /* stochastic gradient descent */
			{
				realdp lossSum = 0;
				gradSum.SetToZeros();
                
                /* permutation the indexall array */ //-- TODO: not use indexall and just use stdindex;
                std::vector<integer> stdindex(Prob->GetDataSize());
                for (integer i = 0; i < Prob->GetDataSize(); i++)
                {
                    stdindex[i] = indexall[i]; //.push_back()
                }
                std::shuffle(stdindex.begin(), stdindex.end(), std::default_random_engine(std::random_device()()));
                for (integer i = 0; i < Prob->GetDataSize();i ++)
                    indexall[i] = stdindex[i];
//                memcpy(result, &stdindex[0], stdindex.size() * sizeof(stdindex[0]));
                stdindex.clear();
                f2 = Prob->Stof(x1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsf++;
				Prob->UpdateEpoch();
//				Prob->BatchIndex = ind;
//				f2 = Prob->f(x1); nf++;
				lossSum += f2;
                Prob->StoGrad(x1, &gf1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsg++;
//                Prob->Grad(x1, &gf1); ng++;
                gradSum.AlphaXaddThis(1, gf1);
                
				for (integer i = 1; i < numBatch; i++)
				{
					Mani->ScalarTimesVector(x1, -stepsize, gf1, &eta2);
					Mani->Retraction(x1, eta2, &x2); nR++;
//					Prob->BatchIndex = ind + i * batchsize;
					f2 = Prob->Stof(x2, indexall.GetSubmatrix(i * BatchSize, (i + 1) * BatchSize - 1, 0, 0)); nsf++;
					lossSum += f2;
                    Prob->StoGrad(x2, &gf2, indexall.GetSubmatrix(i * BatchSize, (i + 1) * BatchSize - 1, 0, 0)); nsg++;
//					Prob->Grad(x2, &gf2); ng++;
					gradSum.AlphaXaddThis(1, gf2);
                    
                    if (std::isnan(f2) || std::isinf(f2)) /*Stop when got a nan or inf*/
                    {
                        printf("New function value is either nan or inf. Stop!\n");
                        break;
                    }
                    /*Switch information at x1 and x2*/
                    xTemp = x1;    x1 = x2; x2 = xTemp;
                    gfTemp = gf1; gf1 = gf2; gf2 = gfTemp;
				}

				f2 = lossSum / numBatch;
				
			}
            
			(this->*StepsizePtr)();
			iter++;
			
			ngf1 = std::sqrt(Mani->Metric(x1, gradSum, gradSum)) / numBatch;
			ngf2 = ngf1;
            
            
            //---------------------------

			if (Verbose >= ITERRESULT)
			{
				/*Output information*/
				if (iter % OutputGap == 0)
					PrintInfo();

				/*Store debug information in the arrays*/
				timeSeriesptr[iter] = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;
				funSeriesptr[iter] = f2;
				gradSeriesptr[iter] = ngf2;
//				xSeries.GetElement(iter) = x1;
			}

			/*Call the function to check whether the stopping criterion is satisfied or not.
			The default function is written in Solvers.h and Solvers.cpp*/
			isstop = IsStopped();
			f1 = f2;
		}

		ComTime = static_cast<realdp>(getTickCount() - starttime) / CLK_PS;

		if (Verbose >= ITERRESULT)
		{
			lengthSeries = iter + 1;
		}
		if (Verbose >= FINALRESULT)
			PrintFinalInfo();
		
//		delete[] ind;
	};

	void RSGD::CheckParams(void)
	{

//		std::string STEPSIZEnames[SELECTSTEPSIZELENGTH] = { "FIXED_STEPSIZE", "STEPLR","EXPLR","INVERSETIMELR" };
		SolversSMSto::CheckParams();
//		char YES[] = "YES";
//		char NO[] = "NO";
		char *status;
		printf("RSGD PARAMETERS: NONE\n");
//		status = (InitStepsize > 0) ? YES : NO;
//		printf("InitStepsize:  %15g[%s],\t", InitStepsize, status);
//		status = (theta >= 0 && theta <= 1) ? YES : NO;
//		printf("theta:         %15g[%s],\n", theta, status);
//		status = (gamma1 > 0 && gamma1 < 1) ? YES : NO;
//		printf("gamma1:        %15g[%s],\t", gamma1, status);
//		status = (gamma2 > 0 && gamma2 < 1) ? YES : NO;
//		printf("gamma2:        %15g[%s],\n", gamma2, status);
//		status = (StepsizeType >= 0 && StepsizeType < SELECTSTEPSIZELENGTH) ? YES : NO;
//		printf("StepsizeType:  %15s[%s],\t", STEPSIZEnames[StepsizeType].c_str(), status);
//        status = (BatchSize >= 1) ? YES : NO;
//        printf("BatchSize:     %15d[%s],\n", BatchSize, status);
	};

//    void RSGD::PrintInfo(void)
//    {
//        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));
//        
//        printf("|gf|:%.3e,t0:%.2e,t:%.2e,time:%.2g,", ngf2, Initstepsize, stepsize, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
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
//    void RSGD::PrintFinalInfo(void)
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

	void RSGD::SetDefaultParams(void)
	{
		SolversSMSto::SetDefaultParams();
		StepsizeType = STO_STEPLR;
//		Initstepsize = static_cast<realdp>(1);
//		theta = static_cast<realdp>(0);
//		gamma1 = static_cast<realdp>(0.2);
//		gamma2 = static_cast<realdp>(0.95);
//        BatchSize = 1;
//		isFixed = true;
//		NumFixedStep = 10;
//		burnin_multiplier = 1;
//        nsf = 0;
//        nsg = 0;
		SolverName.assign("RSGD");
	};

	void RSGD::SetParams(PARAMSMAP params)
	{
		SolversSMSto::SetParams(params);
//		PARAMSMAP::iterator iter;
//		for (iter = params.begin(); iter != params.end(); iter++)
//		{
//			if (iter->first == static_cast<std::string> ("theta"))
//			{
//				theta = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("gamma1"))
//			{
//				gamma1 = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("gamma2"))
//			{
//				gamma2 = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("StepsizeType"))
//			{
//				StepsizeType = static_cast<RSGDLR_SCHEDULER> (static_cast<integer> (iter->second));
//			}
//			else if (iter->first == static_cast<std::string> ("InitStepsize"))
//			{
//				InitStepsize = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("isFixed"))
//			{
//				isFixed = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("NumFixedStep"))
//			{
//				NumFixedStep = iter->second;
//			}
//			else if (iter->first == static_cast<std::string> ("burnin_multiplier"))
//			{
//				burnin_multiplier = iter->second;
//			}
//            else if (iter->first == static_cast<std::string> ("BatchSize"))
//            {
//                BatchSize = iter->second;
//            }
//		}

	};

	void RSGD::SetProbX(const Problem *prob, const Variable *initialx)
	{
		SolversSMSto::SetProbX(prob, initialx);
		prob->SetUseGrad(true);
		prob->SetUseHess(false);
	};

//	void RSGD::ChooseStepsize(void)
//	{
//		if (StepsizeType == FIXED_STEPSIZE)
//			StepsizePtr = &RSGD::FixedStepsize;
//		else if (StepsizeType == STEPLR)
//			StepsizePtr = &RSGD::StepLr;
//		else if (StepsizeType == EXPLR)
//			StepsizePtr = &RSGD::ExpLr;
//		else if (StepsizeType == INVERSETIMELR)
//			StepsizePtr = &RSGD::InverseTimeLr;
//	};
//
//	void RSGD::FixedStepsize(void)
//	{
//		stepsize = (iter <= NumFixedStep) ? (InitStepsize * burnin_multiplier) : InitStepsize;
//	};
//
//	void RSGD::StepLr()
//	{
//		if (isFixed)
//		{
//			stepsize = (iter <= NumFixedStep) ? (InitStepsize * burnin_multiplier) : (InitStepsize / pow(iter + 1, theta));
//		}
//		else
//		{
//			stepsize = InitStepsize / pow(iter + 1, theta);
//		}
//
//	};
//
//	void RSGD::ExpLr(void)
//	{
//		stepsize = pow(gamma2, iter) * InitStepsize;
//	};
//
//	void RSGD::InverseTimeLr(void)
//	{
//		stepsize = InitStepsize / (1 + gamma1 * iter);
//	}
}
