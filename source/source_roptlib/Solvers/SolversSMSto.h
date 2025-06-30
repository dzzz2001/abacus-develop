
/*
Solvers --> SolversSM --> SolversSMSto
 
 A base class for Riemannian stochastic types of methods
 
 by Wen Huang
*/


#ifndef SOLVERSSMSTO_H
#define SOLVERSSMSTO_H
#include "Solvers/SolversSM.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB {
	/* Update the stepsize.
	FIXED_STEPSIZE : The stepsize remains constant.
	STEPLR : lr = InitStepsize / (iter)^theta.
	EXPLR :  lr = pow(gamma2, iter) * InitStepsize
	InverseTimeLr : lr = InitStepsize / (1 + gamma1 * iter)
	*/
	enum StoLRSCHEDULER { STO_FIXED_STEPSIZE, STO_STEPLR, STO_EXPLR, STO_INVERSETIMELR, STOLRSCHEDULERLENGTH };
	class SolversSMSto : public SolversSM {
	public:
		/*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
		This function is used to set the parameters by the mapping*/
		virtual void SetParams(PARAMSMAP params);

		/*Check whether the parameters about Riemannian stochastic type of algorithms are legal or not.*/
		virtual void CheckParams(void);

		/*Select stepsize.
		Default: STEPLR */
        StoLRSCHEDULER StepsizeType;

        /*initial stepsize
        Default: 0.1*/
		realdp Initstepsize;

		/*STEPLR: stepsize = InitStepsize / (iter)^theta
		Default: 0*/
		realdp theta;

		/*INVERSETIMELR: stepsize = InitStepsize /(1 + gamma1 * iter)
		Default: 0.2*/
		realdp gamma1;

		/*EXPLR: stepsize = InitStepsize * gamma2 ^ iter
		Default: 0.95*/
		realdp gamma2;

		/*Whether to fixed the StepSize of the first NumFixedStep steps*/
		bool isFixed;

		/*Number of steps with a fixed StepSize
		Default: 10*/
		integer NumFixedStep;

		/*Burnin period learning rate product factor
		Default: 0.1*/
		realdp burnin_multiplier;
        
        /* batch size, default: 1 */
        integer BatchSize;
        
	protected:
		/*Call Solvers::SetProbX function and indicate RSGD does not need action of Hessian.
		INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams(void);
        
        /*Print information in every few iterations specific to an algorithm*/
        virtual void PrintInfo(void);
        
        /*Print last information in an algorithm*/
        virtual void PrintFinalInfo(void);
        
		/*When one iteration, some algorithms need to update some information. For example,
		quasi-Newton methods need to update the Hessian approximation and nonlinear conjugate gradient
		needs to update the search direction. They are done in the following function*/
		virtual void UpdateData(void){ };

		virtual void ChooseStepsize(void);

		virtual void FixedStepsize(void);

		virtual void StepLr();

		virtual void ExpLr(void);

		virtual void InverseTimeLr(void);
        
        void (SolversSMSto::*StepsizePtr)(void);

        realdp stepsize;
        
        integer nsf;            /*the number of batch function evaluations*/
        integer nsg;            /*the number of batch gradient evaluations*/
	};
};

#endif /* end of RSGD_H */
