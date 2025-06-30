#include "Solvers/RAMSGRAD.h"

/*Define the namespace*/
namespace ROPTLIB {

	RAMSGRAD::RAMSGRAD(const Problem *prob, const Variable *initialx)
	{
		Initialization(prob, initialx);
	}

	void RAMSGRAD::Run(void)
	{
		Variable xTemp(x1);
		Vector gfTemp = Prob->GetDomain()->GetEMPTY();
		Vector gradSum = Prob->GetDomain()->GetEMPTY();
		gradSum.SetToZeros();

		SolversSMSto::Run();

		m.SetToZeros();
		v.SetToZeros();
		vhat.SetToZeros();
		sqrvhat.SetToZeros();

		integer steps = 0;
        integer N = Prob->GetDataSize(); // num
//        integer num = Prob->num; //number of sample
//        integer batchsize = Prob->BatchSize;
        integer numBatch = static_cast<integer>(N / BatchSize); // Batch
//        integer Batch = Prob->GetBatch();
        Vector indexall(N);
        indexall.NewMemoryOnWrite();
        for(integer i = 0; i < N; i++)
            indexall[i] = i;
        
//        integer *ind = new integer[num];
//        integer n = gf1.GetElement(0).Getlength(); //Embedded dimension
//        Prob->GenerateInitialIndex(num, ind);
//        Prob->BatchIndex = ind;
//        Prob->BatchSize = num;
        f1 = Prob->f(x1); nf++;
        f2 = f1;
        Prob->Grad(x1, &gradSum); ng++;
        gf1 = gradSum;
//        Prob->BatchSize = batchsize;
        		

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

		realdp sqrvhat_min = 0, sqrvhat_max = 0;
		bool isstop = IsStopped();

		/*Start the loop*/
		while (((!isstop) && iter < Max_Iteration) || iter < Min_Iteration)
		{
            Prob->burnin = (iter <= NumFixedStep) ? true : false;
			realdp bound = C / pow((iter + 1), gamma);
			if (numBatch == 1) /* full gradient descent */
			{
				Mani->VectorLinearCombination(x1, 1.0 - beta1, gf1, beta1, m, &m); //m <- beta1 * m + (1-beta1) * gf1;
				Mani->VectorLinearCombination(x1, 1.0 - beta2, gf1.GetHadamardProduct(gf1), beta2, v, &v); //v <- beta2 * v + (1-beta2) * (gf1)^2;

				bias_correction1 = 1.0 - pow(beta1, iter + 1);
				bias_correction2 = 1.0 - pow(beta2, iter + 1);

				vhat = v.GetMax(vhat);
				sqrvhat = (vhat / bias_correction2).GetSqrt();

				sqrvhat_min = sqrvhat.Min();
				sqrvhat_max = sqrvhat.Max();

				if (sqrvhat_max - sqrvhat_min > bound)
				{
					sqrvhat = (sqrvhat - sqrvhat_min) / (sqrvhat_max - sqrvhat_min) * bound + sqrvhat_max - bound;
				}
				eta2 = m / bias_correction1;
				eta2 = eta2.GetHadamardDivision(sqrvhat + epsilon);
				Mani->ScalarTimesVector(x1, -stepsize, eta2, &eta2);
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
			else
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
                Prob->UpdateEpoch();
//
//                Prob->UpdateEpoch(ind);
//                Prob->BatchIndex = ind;
//                f2 = Prob->f(x1); nf++;
                f2 = Prob->Stof(x1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsf++;
                lossSum += f2;
//                Prob->Grad(x1, &gf1); ng++;
                Prob->StoGrad(x1, &gf1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsg++;
                gradSum.AlphaXaddThis(1, gf1);
                
                
				for (integer i = 1; i < numBatch; i++)
				{
					steps++;
					Mani->VectorLinearCombination(x1, 1.0 - beta1, gf1, beta1, m, &m); //m <- beta1 * m + (1-beta1) * gf1;
					Mani->VectorLinearCombination(x1, 1.0 - beta2, gf1.GetHadamardProduct(gf1), beta2, v, &v); //v <- beta2 * v + (1-beta2) * (gf1)^2;

					bias_correction1 = 1.0 - pow(beta1, steps);
					bias_correction2 = 1.0 - pow(beta2, steps);

					vhat = v.GetMax(vhat);
					sqrvhat = (vhat / bias_correction2).GetSqrt();

					sqrvhat_min = sqrvhat.Min();
					sqrvhat_max = sqrvhat.Max();

					if (sqrvhat_max - sqrvhat_min > bound)
					{
						sqrvhat = (sqrvhat - sqrvhat_min) / (sqrvhat_max - sqrvhat_min) * bound + sqrvhat_max - bound;
					}

					eta2 = m / bias_correction1;
					eta2 = eta2.GetHadamardDivision(sqrvhat + epsilon);
					Mani->ScalarTimesVector(x1, -stepsize, eta2, &eta2);
					Mani->Retraction(x1, eta2, &x2); nR++;

                    f2 = Prob->Stof(x2, indexall.GetSubmatrix(i * BatchSize, (i + 1) * BatchSize - 1, 0, 0)); nsf++;
//                    Prob->BatchIndex = ind + i * batchsize;
//                    f2 = Prob->f(x2); nf++;
                    lossSum += f2;
                    Prob->StoGrad(x2, &gf2, indexall.GetSubmatrix(i * BatchSize, (i + 1) * BatchSize - 1, 0, 0)); nsg++;
//                    Prob->Grad(x2, &gf2); ng++;
                    gradSum.AlphaXaddThis(1, gf2);
                    
					if (std::isnan(f2) || std::isinf(f2)) /*Stop when got a nan or inf*/
					{
						printf("New function value is either nan or inf. Stop!\n");
						break;
					}

					/*Switch information at x1 and x2*/
					xTemp = x1;	x1 = x2; x2 = xTemp;
					gfTemp = gf1; gf1 = gf2; gf2 = gfTemp;
				}
				
				f2 = lossSum / numBatch;
				
			}

			(this->*StepsizePtr)();
			iter++;
			
			ngf1 = sqrt(Mani->Metric(x1, gradSum, gradSum)) / numBatch;
			ngf2 = ngf1;

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

	void RAMSGRAD::CheckParams(void)
	{
		SolversSMSto::CheckParams();
//		std::string STEPSIZEnames[STEPSIZELENGTH_RAMSGRADLR] = { "FIXED_STEPSIZE", "STEPLR" };
		char YES[] = "YES";
		char NO[] = "NO";
		char *status;
		printf("RAMSGRAD PARAMETERS:\n");
		status = (beta1 >= 0 && beta1 < 1) ? YES : NO;
        printf("beta1:         %15g[%s],\t", beta1, status);
		status = (beta2 >= 0 && beta2 < 1) ? YES : NO;
		printf("beta2:         %15g[%s],\n", beta2, status);
		status = (epsilon >= 0 && epsilon < 1) ? YES : NO;
		printf("epsilon:       %15g[%s],\t", epsilon, status);
//		status = (InitStepsize >= 0) ? YES : NO;
//		printf("InitStepsize:  %15g[%s],\n", InitStepsize, status);
//		status = (StepsizeType >= 0 && StepsizeType < STEPSIZELENGTH_RAMSGRADLR) ? YES : NO;
//		printf("StepsizeType:  %15s[%s],\t", STEPSIZEnames[StepsizeType].c_str(), status);
		status = (C >= 0) ? YES : NO;
		printf("C:             %15g[%s],\n", C, status);
		status = (gamma > theta) ? YES : NO;
		printf("gamma:         %15g[%s],\n", gamma, status);
//		status = (theta >= 0) ? YES : NO;
//		printf("theta:         %15g[%s],\n", theta, status);
//        status = (BatchSize >= 1) ? YES : NO;
//        printf("BatchSize:     %15d[%s],\n", BatchSize, status);
//		status = (lambda >= 0 && lambda <= 1) ? YES : NO;
//		printf("lambda:        %15g[%s],\t", lambda, status);

	};

	void RAMSGRAD::SetDefaultParams(void)
	{
		SolversSMSto::SetDefaultParams();
		beta1 = static_cast<realdp>(0.9);
		beta2 = static_cast<realdp>(0.999);
		epsilon = static_cast<realdp>(1e-8);
		Initstepsize = static_cast<realdp>(0.1);
		C = static_cast<realdp>(50);
//		lambda = static_cast<realdp>(0.99);
		gamma = static_cast<realdp>(1);
		theta = static_cast<realdp>(0.5);
		isFixed = true;
		NumFixedStep = 10;
		StepsizeType = STO_STEPLR;
		burnin_multiplier = 1;
        BatchSize = 1;
        nsf = 0;
        nsg = 0;
		SolverName.assign("RAMSGRAD");

	};

//    void RAMSGRAD::PrintInfo(void)
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
//    void RAMSGRAD::PrintFinalInfo(void)
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

	void RAMSGRAD::SetParams(PARAMSMAP params)
	{
		SolversSMSto::SetParams(params);
		PARAMSMAP::iterator iter;
		for (iter = params.begin(); iter != params.end(); iter++)
		{
			if (iter->first == static_cast<std::string> ("beta1"))
			{
				beta1 = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("beta2"))
			{
				beta2 = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("epsilon"))
			{
				epsilon = iter->second;
			}
//			else if (iter->first == static_cast<std::string> ("InitStepsize"))
//			{
//				InitStepsize = iter->second;
//			}
			else if (iter->first == static_cast<std::string> ("C"))
			{
				C = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("gamma"))
			{
				gamma = iter->second;
			}
//			else if (iter->first == static_cast<std::string> ("StepsizeType"))
//			{
//				StepsizeType = static_cast<RAMSGRADLR_SCHEDULER> (static_cast<integer> (iter->second));
//			}
//			else if (iter->first == static_cast<std::string> ("theta"))
//			{
//				theta = iter->second;
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
		}

	};

	void RAMSGRAD::SetProbX(const Problem *prob, const Variable *initialx)
	{
		SolversSMSto::SetProbX(prob, initialx);
		prob->SetUseGrad(true);
		prob->SetUseHess(false);
        if(Prob->GetDomain()->GetIsIntrinsic())
        {
            m = Prob->GetDomain()->GetEMPTYINTR();
            v = Prob->GetDomain()->GetEMPTYINTR();
            vhat = Prob->GetDomain()->GetEMPTYINTR();
            sqrvhat = Prob->GetDomain()->GetEMPTYINTR();
        } else
        {
            m = Prob->GetDomain()->GetEMPTYEXTR();
            v = Prob->GetDomain()->GetEMPTYEXTR();
            vhat = Prob->GetDomain()->GetEMPTYEXTR();
            sqrvhat = Prob->GetDomain()->GetEMPTYEXTR();
        }
	};

//	realdp RAMSGRAD::DecayStepsize()
//	{
//		return InitStepsize / pow(iter + 1,theta);
//	};
//
//	void RAMSGRAD::ScheduleBeta1(realdp &result)
//	{
//		result = result * lambda;
//	};
//
//	void RAMSGRAD::ChooseStepsize(void)
//	{
//		if (StepsizeType == FIXED)
//			StepsizePtr = &RAMSGRAD::FixedStepsize;
//		else if (StepsizeType == DECAY)
//			StepsizePtr = &RAMSGRAD::StepLr;
//	};
//
//	void RAMSGRAD::FixedStepsize(void)
//	{
//		stepsize = (iter <= NumFixedStep) ? (InitStepsize * burnin_multiplier) : InitStepsize;
//	};
//
//	void RAMSGRAD::StepLr()
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
}
