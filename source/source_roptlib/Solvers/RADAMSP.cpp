#include "Solvers/RADAMSP.h"

/*Define the namespace*/
namespace ROPTLIB {

	RADAMSP::RADAMSP(const Problem *prob, const Variable *initialx)
	{
		Initialization(prob, initialx);
	}

	void RADAMSP::Run(void)
	{
		ProductManifold *ProdMani = dynamic_cast<ProductManifold *>(const_cast<Manifold *>(Mani));
        const Manifold *mani = nullptr;
        if(IsProdMani)
            mani = ProdMani->GetManifold(0);
        else
            mani = Mani;
        
		Variable xTemp(x1);
		Vector gfTemp = Prob->GetDomain()->GetEMPTY();
		Vector gradSum = Prob->GetDomain()->GetEMPTY();
		gradSum.SetToZeros();

		SolversSMSto::Run();

		m.SetToZeros();
		v.SetToZeros();
		Vector m1(m.GetElement(0));
		m1.SetToZeros();
		Vector v1(v.GetElement(0));
		v1.SetToZeros();

		integer steps = 0;
        integer N = Prob->GetDataSize(); // num
//		integer num = Prob->num; //number of sample
//		integer batchsize = Prob->BatchSize;
        integer numBatch = static_cast<integer>(N / BatchSize); // Batch
//		integer Batch = Prob->GetBatch();
        Vector indexall(N);
        indexall.NewMemoryOnWrite();
        for(integer i = 0; i < N; i++)
            indexall[i] = i;
//		integer *ind = new integer[num];
//		integer n = gf1.GetElement(0).Getlength(); //Embedded dimension
//		Prob->GenerateInitialIndex(num, ind);
//		Prob->BatchIndex = ind;
//		Prob->BatchSize = num; 
		f1 = Prob->f(x1); nf++;
		f2 = f1;
		Prob->Grad(x1, &gradSum); ng++;
		gf1 = gradSum;
//		Prob->BatchSize = batchsize;

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
			Prob->burnin = (iter <= NumFixedStep) ? true : false;

			if (numBatch == 1) /* full gradient descent */
			{
				Mani->VectorLinearCombination(x1, 1.0 - beta1, gf1, beta1, m, &m); //m <- beta1 * m + (1-beta1) * gf1;
				Mani->VectorLinearCombination(x1, 1.0 - beta2, gf1.GetHadamardProduct(gf1), beta2, v, &v); //v <- beta2 * v + (1-beta2) * (gf1)^2;

				bias_correction1 = 1.0 - pow(beta1, iter + 1);
				bias_correction2 = 1.0 - pow(beta2, iter + 1);

				v1 = v / bias_correction2;
				eta2 = m / bias_correction1;
				eta2 = eta2.GetHadamardDivision(v1.GetSqrt() + epsilon);
				Mani->ScalarTimesVector(x1, -stepsize, eta2, &eta2);
                if(IsProdMani)
                {
                    Mani->Retraction(x1, eta2, &x2); nR++;
                } else
                {
                    Mani->Retraction(x1, eta2.GetElement(0), &x2); nR++;
                }

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
                
//				Prob->UpdateEpoch(ind);
//				Prob->BatchIndex = ind;
//				f2 = Prob->f(x1); nf++;
                f2 = Prob->Stof(x1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsf++;
				lossSum += f2;
//				Prob->Grad(x1, &gf1); ng++;
                Prob->StoGrad(x1, &gf1, indexall.GetSubmatrix(0, BatchSize - 1, 0, 0)); nsg++;
				gradSum.AlphaXaddThis(1, gf1);
                
				for (integer i = 1; i < numBatch; i++)
				{
					steps++;
                    eta2.SetToZeros();
                    bias_correction1 = 1.0 - pow(beta1, steps);
                    bias_correction2 = 1.0 - pow(beta2, steps);
                    if(x1.FieldsExist("flag")) /* if a "flag" vector is stored in x1, then use it as a label to check if a component in x1 needs be updated */
                    {
                        Vector flag = x1.Field("flag");
                        const realdp *flagptr = flag.ObtainReadData();
                        
                        if(IsProdMani)
                        {
                            for (integer j = 0; j < flag.Getlength(); j++)
                            {
                                if (abs(flagptr[j]) < 1e-8)
                                    continue;
                                
                                //m <- beta1 * m + (1-beta1) * gf1
                                mani->VectorLinearCombination(x1.GetElement(j), 1.0 - beta1, gf1.GetElement(j), beta1, m.GetElement(j), &m.GetElement(j));
                                //v <- beta2 * v + (1-beta2) * (gf1)^2
                                mani->VectorLinearCombination(x1.GetElement(j), 1.0 - beta2, gf1.GetElement(j).GetHadamardProduct(gf1.GetElement(j)), beta2, v.GetElement(j), &(v.GetElement(j)));
                                
                                mani->ScalarTimesVector(x1.GetElement(j), 1.0 / bias_correction1, m.GetElement(j), &m1);
                                mani->ScalarTimesVector(x1.GetElement(j), 1.0 / bias_correction2, v.GetElement(j), &v1);
                                mani->ScalarTimesVector(x1.GetElement(j), -stepsize, m1, &(eta2.GetElement(j)));
                                eta2.GetElement(j) = eta2.GetElement(j).GetHadamardDivision(v1.GetSqrt() + epsilon);
                            }
                        } else
                        {
                            //m <- beta1 * m + (1-beta1) * gf1
                            mani->VectorLinearCombination(x1, 1.0 - beta1, gf1, beta1, m, &m);
                            //v <- beta2 * v + (1-beta2) * (gf1)^2
                            mani->VectorLinearCombination(x1, 1.0 - beta2, gf1.GetHadamardProduct(gf1), beta2, v, &v);
                            
                            mani->ScalarTimesVector(x1, 1.0 / bias_correction1, m, &m1);
                            mani->ScalarTimesVector(x1, 1.0 / bias_correction2, v, &v1);
                            mani->ScalarTimesVector(x1, -stepsize, m1, &eta2);
                            eta2 = eta2.GetHadamardDivision(v1.GetSqrt() + epsilon);
                        }
                    } else /* if no "flag" vector label is created, then compute the gradient of each component to see if it has been touched. This approach is more computationally expensive
                            but easier to use in practice.*/
                    {
                        if(IsProdMani)
                        {
                            for (integer j = 0; j < gf1.Getnumofelements(); j++)
                            {
                                if(gf1.GetElement(j).Fnorm() <= std::numeric_limits<realdp>::epsilon()) /* if this component has not been touched by the chosen data, then skip the update */
                                    continue;
                                
                                //m <- beta1 * m + (1-beta1) * gf1
                                mani->VectorLinearCombination(x1.GetElement(j), 1.0 - beta1, gf1.GetElement(j), beta1, m.GetElement(j), &m.GetElement(j));
                                //v <- beta2 * v + (1-beta2) * (gf1)^2
                                mani->VectorLinearCombination(x1.GetElement(j), 1.0 - beta2, gf1.GetElement(j).GetHadamardProduct(gf1.GetElement(j)), beta2, v.GetElement(j), &(v.GetElement(j)));
                                
                                mani->ScalarTimesVector(x1.GetElement(j), 1.0 / bias_correction1, m.GetElement(j), &m1);
                                mani->ScalarTimesVector(x1.GetElement(j), 1.0 / bias_correction2, v.GetElement(j), &v1);
                                mani->ScalarTimesVector(x1.GetElement(j), -stepsize, m1, &(eta2.GetElement(j)));
                                eta2.GetElement(j) = eta2.GetElement(j).GetHadamardDivision(v1.GetSqrt() + epsilon);
                            }
                        } else
                        {
                            //m <- beta1 * m + (1-beta1) * gf1
                            mani->VectorLinearCombination(x1, 1.0 - beta1, gf1, beta1, m, &m);
                            //v <- beta2 * v + (1-beta2) * (gf1)^2
                            mani->VectorLinearCombination(x1, 1.0 - beta2, gf1.GetHadamardProduct(gf1), beta2, v, &v);
                            mani->ScalarTimesVector(x1, 1.0 / bias_correction1, m.GetElement(0), &m1);
                            mani->ScalarTimesVector(x1, 1.0 / bias_correction2, v.GetElement(0), &v1);
                            mani->ScalarTimesVector(x1, -stepsize, m1, &eta2);

                            eta2 = eta2.GetHadamardDivision(v1.GetSqrt() + epsilon);
                        }
                    }
                    
//                    x1.Print("x1:");//--
//                    eta2.Print("eta2:");//--
//                    x2.Print("x2:");//---
                    
					Mani->Retraction(x1, eta2, &x2); nR++;

                    f2 = Prob->Stof(x2, indexall.GetSubmatrix(i * BatchSize, (i + 1) * BatchSize - 1, 0, 0)); nsf++;
//					Prob->BatchIndex = ind + i * batchsize;
//					f2 = Prob->f(x2); nf++;
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
					xTemp = x1;	x1 = x2; x2 = xTemp;
					gfTemp = gf1; gf1 = gf2; gf2 = gfTemp;
				}

				f2 = lossSum / numBatch;

			}

			(this->*StepsizePtr)();
			iter++;

			ngf1 = std::sqrt(Mani->Metric(x1, gradSum, gradSum)) / numBatch;
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

	void RADAMSP::CheckParams(void)
	{
//		std::string STEPSIZEnames[STEPSIZELENGTH_RADAMSPLR] = { "FIXED", "STEPLR" };
		SolversSMSto::CheckParams();
		char YES[] = "YES";
		char NO[] = "NO";
		char *status;
		printf("RADAMSP PARAMETERS:\n");
		status = (beta1 >= 0 && beta1 < 1) ? YES : NO;
		printf("beta1:         %15g[%s],\t", beta1, status);
		status = (beta2 >= 0 && beta2 < 1) ? YES : NO;
		printf("beta2:         %15g[%s],\n", beta2, status);
		status = (epsilon >= 0 && epsilon < 1) ? YES : NO;
		printf("epsilon:       %15g[%s]\n", epsilon, status);
//		status = (InitStepsize >= 0) ? YES : NO;
//		printf("InitStepsize:  %15g[%s],\n", InitStepsize, status);
//		status = (StepsizeType >= 0 && StepsizeType < STEPSIZELENGTH_RADAMSPLR) ? YES : NO;
//		printf("StepsizeType:  %15s[%s],\t", STEPSIZEnames[StepsizeType].c_str(), status);
//		status = (theta >= 0) ? YES : NO;
//		printf("theta:         %15g[%s],\n", theta, status);
//        status = (BatchSize >= 1) ? YES : NO;
//        printf("BatchSize:     %15d[%s],\n", BatchSize, status);
////		status = (lambda >= 0 && lambda <= 1) ? YES : NO;
////		printf("lambda:%15g[%s],\t", lambda, status);
	};

	void RADAMSP::SetDefaultParams(void)
	{
		SolversSMSto::SetDefaultParams();
		beta1 = static_cast<realdp>(0.9);
		beta2 = static_cast<realdp>(0.999);
		epsilon = static_cast<realdp>(1e-8);
		Initstepsize = static_cast<realdp>(0.001);
		stepsize = Initstepsize;
//		isFixed = true;
//		NumFixedStep = 10;
//		theta = static_cast<realdp>(0);
//		lambda = static_cast<realdp>(0.99);
		StepsizeType = STO_FIXED_STEPSIZE;
		burnin_multiplier = 0.1;
//        nsf = 0;
//        nsg = 0;
//        BatchSize = 1;
		SolverName.assign("RADAMSP");
	};

//    void RADAMSP::PrintInfo(void)
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
//    void RADAMSP::PrintFinalInfo(void)
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

	void RADAMSP::SetParams(PARAMSMAP params)
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
////			else if (iter->first == static_cast<std::string> ("lambda"))
////			{
////				lambda = iter->second;
////			}
//			else if (iter->first == static_cast<std::string> ("StepsizeType"))
//			{
//				StepsizeType = static_cast<RADAMSPLR_SCHEDULER> (static_cast<integer> (iter->second));
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

	void RADAMSP::SetProbX(const Problem *prob, const Variable *initialx)
	{
		SolversSMSto::SetProbX(prob, initialx);
        IsProdMani = (dynamic_cast<ProductManifold *>(const_cast<Manifold *>(Prob->GetDomain())) == nullptr) ? false : true;
		prob->SetUseGrad(true);
		prob->SetUseHess(false);
        if(IsProdMani)
        {
            if(Prob->GetDomain()->GetIsIntrinsic())
            {
                m = Prob->GetDomain()->GetEMPTYINTR();
                v = Prob->GetDomain()->GetEMPTYINTR();
            } else
            {
                m = Prob->GetDomain()->GetEMPTYEXTR();
                v = Prob->GetDomain()->GetEMPTYEXTR();
            }
        } else
        {
            Vector mm, vv;
            if(Prob->GetDomain()->GetIsIntrinsic())
            {
                mm = Prob->GetDomain()->GetEMPTYINTR();
                vv = Prob->GetDomain()->GetEMPTYINTR();
            } else
            {
                mm = Prob->GetDomain()->GetEMPTYEXTR();
                vv = Prob->GetDomain()->GetEMPTYEXTR();
            }
            Variable mmm(1, &mm, 1);
            Variable vvv(1, &vv, 1);
            m = mmm;
            v = vvv;
//            Variable StieX(3, 2), EucX(2);
//            Variable ProdX(2, &StieX, 2, &EucX, 1);
        }
	};

//	void RADAMSP::ScheduleBeta1(realdp &result)
//	{
//		result = result * lambda;
//	};
//
//	void RADAMSP::ChooseStepsize(void)
//	{
//		if (StepsizeType == FIXED_RADAMSPLR)
//			StepsizePtr = &RADAMSP::FixedStepsize;
//		else if (StepsizeType == DECAY_RADAMSPLR)
//			StepsizePtr = &RADAMSP::StepLr;
//	};
//
//	void RADAMSP::FixedStepsize(void)
//	{
//		stepsize = (iter <= NumFixedStep) ? (InitStepsize * burnin_multiplier) : InitStepsize;
//	};
//
//	void RADAMSP::StepLr()
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
