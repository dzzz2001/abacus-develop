#include "Solvers/SolversSMSto.h"

/*Define the namespace*/
namespace ROPTLIB {

	void SolversSMSto::CheckParams(void)
	{

		std::string STEPSIZEnames[STOLRSCHEDULERLENGTH] = { "STO_FIXED_STEPSIZE", "STO_STEPLR","STO_EXPLR","STO_INVERSETIMELR" };
		SolversSM::CheckParams();
		char YES[] = "YES";
		char NO[] = "NO";
		char *status;
		printf("SolversSMSto PARAMETERS:\n");
		status = (Initstepsize > 0) ? YES : NO;
		printf("Initstepsize:  %15g[%s],\t", Initstepsize, status);
		status = (theta >= 0) ? YES : NO;
		printf("theta:         %15g[%s],\n", theta, status);
		status = (gamma1 > 0) ? YES : NO;
		printf("gamma1:        %15g[%s],\t", gamma1, status);
		status = (gamma2 > 0 && gamma2 < 1) ? YES : NO;
		printf("gamma2:        %15g[%s],\n", gamma2, status);
		status = (StepsizeType >= 0 && StepsizeType < STOLRSCHEDULERLENGTH) ? YES : NO;
		printf("StepsizeType:  %15s[%s],\t", STEPSIZEnames[StepsizeType].c_str(), status);
        status = (BatchSize >= 1) ? YES : NO;
        printf("BatchSize:     %15d[%s],\n", BatchSize, status);
        printf("isFixed:       %15d[%s],\t", isFixed, status);
        status = (NumFixedStep > 0) ? YES : NO;
        printf("NumFixedStep:  %15d[%s],\n", NumFixedStep, status);
        status = (burnin_multiplier > 0) ? YES : NO;
        printf("burnin_multiplier: %15g[%s],\n", burnin_multiplier, status);
	};

    void SolversSMSto::PrintInfo(void)
    {
        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));
        
        printf("|gf|:%.3e,t0:%.2e,t:%.2e,time:%.2g,", ngf2, Initstepsize, stepsize, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        
        printf("nf:%d,ng:%d,nsf:%d,nsg:%d,", nf, ng, nsf, nsg);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
    };

    void SolversSMSto::PrintFinalInfo(void)
    {
        printf("i:%d,f:%.3e,", iter, f2);
        
        printf("|gf|:%.3e,|gf|/|gf0|:%.3e,time:%.2g,", ngf2, ngf2/ngf0, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        
        printf("nf:%d,ng:%d,nsf:%d,nsg:%d,", nf, ng, nsf, nsg);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
    };

	void SolversSMSto::SetDefaultParams(void)
	{
		SolversSM::SetDefaultParams();
		StepsizeType = STO_FIXED_STEPSIZE;
		Initstepsize = static_cast<realdp>(1);
		theta = static_cast<realdp>(0);
		gamma1 = static_cast<realdp>(0.2);
		gamma2 = static_cast<realdp>(0.95);
        BatchSize = 1;
		isFixed = true;
		NumFixedStep = 10;
		burnin_multiplier = 1;
        nsf = 0;
        nsg = 0;
	};

	void SolversSMSto::SetParams(PARAMSMAP params)
	{
		SolversSM::SetParams(params);
		PARAMSMAP::iterator iter;
		for (iter = params.begin(); iter != params.end(); iter++)
		{
			if (iter->first == static_cast<std::string> ("theta"))
			{
				theta = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("gamma1"))
			{
				gamma1 = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("gamma2"))
			{
				gamma2 = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("StepsizeType"))
			{
				StepsizeType = static_cast<StoLRSCHEDULER> (static_cast<integer> (iter->second));
			}
			else if (iter->first == static_cast<std::string> ("Initstepsize"))
			{
				Initstepsize = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("isFixed"))
			{
				isFixed = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("NumFixedStep"))
			{
				NumFixedStep = iter->second;
			}
			else if (iter->first == static_cast<std::string> ("burnin_multiplier"))
			{
				burnin_multiplier = iter->second;
			}
            else if (iter->first == static_cast<std::string> ("BatchSize"))
            {
                BatchSize = iter->second;
            }
		}

	};

	void SolversSMSto::SetProbX(const Problem *prob, const Variable *initialx)
	{
		SolversSM::SetProbX(prob, initialx);
		prob->SetUseGrad(true);
		prob->SetUseHess(false);
	};

	void SolversSMSto::ChooseStepsize(void)
	{
		if (StepsizeType == STO_FIXED_STEPSIZE)
			StepsizePtr = &SolversSMSto::FixedStepsize;
		else if (StepsizeType == STO_STEPLR)
			StepsizePtr = &SolversSMSto::StepLr;
		else if (StepsizeType == STO_EXPLR)
			StepsizePtr = &SolversSMSto::ExpLr;
		else if (StepsizeType == STO_INVERSETIMELR)
			StepsizePtr = &SolversSMSto::InverseTimeLr;
	};

	void SolversSMSto::FixedStepsize(void)
	{
//        std::cout << "stepsize:" << stepsize << ", Initstepsize:" << Initstepsize << std::endl;//---
		stepsize = (iter <= NumFixedStep) ? (Initstepsize * burnin_multiplier) : Initstepsize;
//        std::cout << "stepsize:" << stepsize << std::endl;//--
	};

	void SolversSMSto::StepLr()
	{
		if (isFixed)
		{
			stepsize = (iter <= NumFixedStep) ? (Initstepsize * burnin_multiplier) : (Initstepsize / pow(iter + 1, theta));
		}
		else
		{
			stepsize = Initstepsize / pow(iter + 1, theta);
		}

	};

	void SolversSMSto::ExpLr(void)
	{
		stepsize = pow(gamma2, iter) * Initstepsize;
	};

	void SolversSMSto::InverseTimeLr(void)
	{
		stepsize = Initstepsize / (1 + gamma1 * iter);
	}
}
