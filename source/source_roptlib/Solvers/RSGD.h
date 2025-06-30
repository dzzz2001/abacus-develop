
/*
Solvers --> SolversSM --> SolversSMSto --> RSGD
 
 A Riemannian stochastic gradient method
 
 written by Yihui Huang, modified by Wen Huang
*/


#ifndef RSGD_H
#define RSGD_H
#include "Solvers/SolversSMSto.h"
#include "Others/def.h"
#include <algorithm>

/*Define the namespace*/
namespace ROPTLIB {
	class RSGD : public SolversSMSto {
	public:
		/*The contructor of RSGD method. It calls the function Solvers::Initialization.
		INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		RSGD(const Problem *prob, const Variable *initialx);

		/*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
		This function is used to set the parameters by the mapping*/
		virtual void SetParams(PARAMSMAP params);

		/*Check whether the parameters about RSGD algorithms are legal or not.*/
		virtual void CheckParams(void);

		void Run(void);

//		/*Select stepsize.
//		Default: STEPLR */
//		RSGDLR_SCHEDULER StepsizeType;

//        /*initial stepsize
//        Default: 0.1*/
//		realdp InitStepsize;

//		/*STEPLR: stepsize = InitStepsize / (iter)^theta
//		Default: 0*/
//		realdp theta;
//
//		/*INVERSETIMELR: stepsize = InitStepsize /(1 + gamma1 * iter)
//		Default: 0.2*/
//		realdp gamma1;
//
//		/*EXPLR: stepsize = InitStepsize * gamma2 ^ iter
//		Default: 0.95*/
//		realdp gamma2;
//
//		/*Gradients for all samples*/
//		//Vector gf_all;
//
//		/*Whether to fixed the StepSize of the first NumFixedStep steps*/
//		bool isFixed;
//
//		/*Number of steps with a fixed StepSize
//		Default: 10*/
//		integer NumFixedStep;
//
//		/*Burnin period learning rate product factor
//		Default: 0.1*/
//		realdp burnin_multiplier;
//        
//        /* batch size, default: 1 */
//        integer BatchSize;
        
	protected:
		/*Call Solvers::SetProbX function and indicate RSGD does not need action of Hessian.
		INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams(void);
        
//        /*Print information in every few iterations specific to an algorithm*/
//        virtual void PrintInfo(void);
//        
//        /*Print last information in an algorithm*/
//        virtual void PrintFinalInfo(void);
        
		/*When one iteration, some algorithms need to update some information. For example,
		quasi-Newton methods need to update the Hessian approximation and nonlinear conjugate gradient
		needs to update the search direction. They are done in the following function*/
		virtual void UpdateData(void){ };

//		void (RSGD::*StepsizePtr)(void);
//
//		virtual void ChooseStepsize(void);
//
//		virtual void FixedStepsize(void);
//
//		virtual void StepLr();
//
//		virtual void ExpLr(void);
//
//		virtual void InverseTimeLr(void);
//        
//        realdp stepsize;
        
//        integer nsf;            /*the number of batch function evaluations*/
//        integer nsg;            /*the number of batch gradient evaluations*/
	};
};

#endif /* end of RSGD_H */
