/*
 A Riemannian ADAM algorithm, see
   Yihui Huang, Wen Huang, Generalization of Adaptive Optimization Algorithm on Manifolds, 2024
 
Solvers --> SolversSM --> RAMSGRAD
 
 written by Yihui Huang, modified by Wen Huang
*/
#ifndef RAMSGRAD_H
#define RAMSGRAD_H

#include <iostream>
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include "Solvers/SolversSMSto.h"
#include "Others/def.h"
#include <algorithm>

/*Define the namespace*/
namespace ROPTLIB {
//	enum RAMSGRADLR_SCHEDULER { FIXED, DECAY, STEPSIZELENGTH_RAMSGRADLR };
	class RAMSGRAD : public SolversSMSto {
	public:
		RAMSGRAD(const Problem *prob, const Variable *initialx);

		virtual void Run(void);

		/*Check whether the parameters about linesearch algorithms are legal or not.*/
		virtual void CheckParams(void);

		/*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
		This function is used to set the parameters by the mapping*/
		virtual void SetParams(PARAMSMAP params);

		/*Exponential decay rate of first-order moment
		Default: 0.9*/
		realdp beta1;

		/*Exponential decay rate of second-order moment
		Default: 0.999*/
		realdp beta2;

//		/*Stepsize
//		Default: 0.001*/
//		realdp InitStepsize;

		/*Used for numeric stability
		Default: 1e-8*/
		realdp epsilon;

		/*Gradients for all samples*/
		//Vector gf_all;

//		/*Parameters related to beta1: beta1 * lambda^(t - 1) 
//		Default: 0.99*/
//		realdp lambda;

		/*Parameter related to \sqrt(\hat(v)_t): 
		max(\sqrt(\hat(v)_t)) - min(\sqrt(\hat(v)_t)) < C / t^gamma;
		Default: 50*/
		realdp C;

		/*Parameter related to \sqrt(\hat(v)_t):
		max(\sqrt(\hat(v)_t)) - min(\sqrt(\hat(v)_t)) < C / t^gamma;
		Default: 1*/
		realdp gamma;

//		/*Parameter related to stepsize:
//		stepsize = InitStepsize / t^theta;
//		Default: 0.5*/
//		realdp theta;
//
//		/*Whether to fixed the StepSize of the first n steps*/
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
//		/*Select stepsize.
//		Default: FIXED_STEPSIZE */
//		RAMSGRADLR_SCHEDULER StepsizeType;
//
//        /* batch size, default: 1 */
//        integer BatchSize;
	protected:

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams(void);
        
//        /*Print information in every few iterations specific to an algorithm*/
//        virtual void PrintInfo(void);
//        
//        /*Print last information in an algorithm*/
//        virtual void PrintFinalInfo(void);
        
		/*Call Solvers::SetProbX function and indicate AMSGRAD does not need action of Hessian.
		INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);

		/*When one iteration, some algorithms need to update some information. For example,
		quasi-Newton methods need to update the Hessian approximation and nonlinear conjugate gradient
		needs to update the search direction. They are done in the following function*/
		virtual void UpdateData(void) { };

//		virtual realdp DecayStepsize();

//		virtual void ScheduleBeta1(realdp &result);
//
//		void (RAMSGRAD::*StepsizePtr)(void);
//
//		virtual void ChooseStepsize(void);
//
//		virtual void FixedStepsize(void);
//
//		virtual void StepLr();

		/* algorithm-related variables: */
		Vector m; /*First moment estimation*/
		Vector v; /*Second moment estimation*/
		Vector vhat;
		Vector sqrvhat;
		realdp bias_correction1;
		realdp bias_correction2;

//		realdp stepsize;
//        
//        integer nsf;            /*the number of batch function evaluations*/
//        integer nsg;            /*the number of batch gradient evaluations*/
	};
};/*end of ROPTLIB namespace*/

#endif /* end of RAMSGRAD_H */
