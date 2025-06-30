
/*
 A Riemannian ADAM algorithm with sparse update on a product of manifolds, see
   Yihui Huang, Wen Huang, Generalization of Adaptive Optimization Algorithm on Manifolds, 2024
 
 All the component of the product of manifolds are the same, i.e., \mathcal{M}^n
 The objective function f = \sum_{i = 1}^N f_i, where f_i:\mathcal{M}^n \rightarrow \mathbb{R}: (x_1, x_2, \ldots, x_n) \mapsto f_i(x_1, x_2, \ldots, x_n).
 For any $i$, f_i does not depend on all $x_j, j = 1, \ldots, n$. f_i only depends on very few $x_j$. Therefore, in the stochastic gradient-type algorithm,
 if f_i is chosen, then only x_j, that influences f_i, and the corresponding first/second order moment are updated.
 
 Solvers --> SolversSM --> SolversSMSto --> RADAMSP
 
 by Yihui Huang, modified by Wen Huang
*/

#ifndef RADAMSP_H
#define RADAMSP_H

#include <iostream>
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include "Solvers/SolversSMSto.h"
#include "Others/def.h"
#include "Others/Timer.h"
#include "Manifolds/MultiManifolds.h"
#include <algorithm>

/*Define the namespace*/
namespace ROPTLIB {
//	enum RADAMSPLR_SCHEDULER { FIXED_RADAMSPLR, DECAY_RADAMSPLR, STEPSIZELENGTH_RADAMSPLR};
	class RADAMSP : public SolversSMSto {
	public:
		RADAMSP(const Problem *prob, const Variable *initialx);

		virtual void Run(void);

//		/*Select stepsize.
//		Default: STEPLR */
//		RADAMSPLR_SCHEDULER StepsizeType;

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

//		/*Gradients for all samples*/
//		//Vector gf_all;
//
//		/*Parameter related to stepsize:
//		stepsize = InitStepsize / iter^theta;
//		Default: 0*/
//		realdp theta;
//
//		/*Whether to fixed the StepSize of the first NumFixedStep steps*/
//		bool isFixed;
//
//		/*Number of steps with a fixed StepSize
//		Default: 10*/
//		integer NumFixedStep;
//
////		/*Parameters related to beta1: beta1 * lambda^(t - 1)
////		Default: 0.99*/
////		realdp lambda;
//
//		/*Burnin period learning rate product factor
//		Default: 0.1*/
//		realdp burnin_multiplier;
//        
//        /* batch size, default: 1 */
//        integer BatchSize;
	protected:

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams(void);

		/*Call Solvers::SetProbX function and indicate ADAM does not need action of Hessian.
		INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);
        
//        /*Print information in every few iterations specific to an algorithm*/
//        virtual void PrintInfo(void);
//        
//        /*Print last information in an algorithm*/
//        virtual void PrintFinalInfo(void);
        
		/*When one iteration, some algorithms need to update some information. For example,
		quasi-Newton methods need to update the Hessian approximation and nonlinear conjugate gradient
		needs to update the search direction. They are done in the following function*/
		virtual void UpdateData(void) { };
//
//		void (RADAMSP::*StepsizePtr)(void);
//
////		void ScheduleBeta1(realdp &result);
//
//		virtual void ChooseStepsize(void);
//
//		virtual void FixedStepsize(void);
//
//		virtual void StepLr();
//
//		realdp stepsize;

		/* algorithm-related variables: */
		Vector m, v;
		realdp bias_correction1;
		realdp bias_correction2;
        
//        integer nsf;            /*the number of batch function evaluations*/
//        integer nsg;            /*the number of batch gradient evaluations*/
    private:
        bool IsProdMani;
	};
};/*end of ROPTLIB namespace*/

#endif /* end of RADAMSP_H */
