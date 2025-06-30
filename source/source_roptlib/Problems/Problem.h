/*
This file defines the abstract base class for problem classes. All the problem classes
should be derived from this class.

Problem

---- WH
*/

#ifndef PROBLEM_H
#define PROBLEM_H

#include "Manifolds/Element.h"
#include "Manifolds/Manifold.h"
#include <iostream>
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	/*Declaration of Manifold*/
	class Manifold;
    class Problem;

    extern Vector MinMaxEigValHessian(Variable *X, Manifold *Domain, const Problem *Prob);
    extern unsigned long starttime;    /*the start time of running the algorithm*/

	class Problem{
	public:
		/*This function indicates this is an abstract class.*/
		virtual ~Problem(void) = 0;

		/*Evaluate the cost function at iterate x. It must be overloaded by derived class*/
		virtual realdp f(const Variable &x) const = 0;
        
        /*If the objective is in the form of F = f + g, where f is continuously differentiable, g is nonsmooth,
         then the function g is defined here. The function f in f + g is defined in the above function "virtual realdp f(const Variable &x) const = 0;"
         Otherwise, g is a zero function
         Default: g is a zero function*/
        virtual realdp g(const Variable &x) const;

		/* It calls "RieGrad" and obtain the extrinsic representation of the Riemannian gradient.
		 After that this function may or may not convert the extrinsic representation to intrinsic representation
		 based on the "IsIntrApproach" in the domain manifold.
         */
		virtual Vector &Grad(const Variable &x, Vector *result) const;

		/*Compute the action of the Riemannian Hessian of the cost function at iterate x, i.e., xix = Hess f(x) [etax].
		 It calls "RieHess" and obtain the extrinsic representation of the Action of the Hessian.
		 After that this function may or may not convert the extrinsic representation to intrinsic representation
		 based on the "IsIntrApproach" in the domain manifold. */
		virtual Vector &HessianEta(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the Riemanian  gradient of the cost function at iterate x.
		 User can override this function. Otherwise, this function will call "EucGrad" to
		 obtain the Euclidean gradient. Then convert the Euclidean gradient to Riemannian gradient.*/
		virtual Vector &RieGrad(const Variable &x, Vector *result) const;

		/*Compute the action of the Riemanian Hessian of the cost function at iterate x.
		 User can override this function. Otherwise, this function will call "EucHessianEta" to
		 obtain the Euclidean action of the Hessian. Then convert the Euclidean action of the Hessian
		 to Riemannian action of the Hessian. */
		virtual Vector &RieHessianEta(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the Euclidean gradient of f*/
		virtual Vector &EucGrad(const Variable &x, Vector *result) const;

		/*Compute the action of the Euclidean Hessian */
		virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;

		/*The preconditioner.
        Default: result = eta, no preconditioner*/
		virtual Vector &PreConditioner(const Variable &x, const Vector &eta, Vector *result) const;
        
        /*Evaluate the cost function at iterate x for a batch. Note that this function is used for optimization problem \min_x \sum_{i = 1}^N f_i(x)*/
        virtual realdp Stof(const Variable &x, const Vector &batch_index) const {return 0;};
        
        /*Stochastic problem size N. Note that this function is used for optimization problem \min_x \sum_{i = 1}^N f_i(x)*/
        virtual integer GetDataSize(void) const {return 1;};
        
        /* It calls "RieStoGrad" and obtain the extrinsic representation of the Riemannian gradient.
         After that this function may or may not convert the extrinsic representation to intrinsic representation
         based on the "IsIntrApproach" in the domain manifold.
         This is used for stochastic optimization algorithms for problems in the form of \min_x \sum_{i = 1}^N f_i(x), the last input specifies
         the batchsize and the index in one iteration.*/
        virtual Vector &StoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;

        /*Compute the Riemanian gradient of the batch of the function at iterate x.
         User can override this function. Otherwise, this function will call "EucStoGrad" to
         obtain the Euclidean gradient. Then convert the Euclidean gradient to Riemannian gradient.
         This is used for stochastic optimization algorithms for problems in the form of \min_x \sum_{i = 1}^N f_i(x), the last input specifies
         the batchsize and the index in one iteration.*/
        virtual Vector &RieStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;

        /*Compute the Euclidean gradient of the batch of the function f; 
         This is used for stochastic optimization algorithms for problems in the form of \min_x \sum_{i = 1}^N f_i(x), the last input specifies
         the batchsize and the index in one iteration.*/
        virtual Vector &EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;
        
        /*Update the information for each epoch in Riemannian stochastic gradient type algorithms*/
        virtual void UpdateEpoch(void) const { };
        
		/*Proximal mapping:
		     argmin_{y \in R^n} 0.5 \|y-x\|_W^2 + lambda \|y\|_1,
		 where W is a diagonal weights.
         Default: result = x, i.e., lambda = 0*/
		virtual Vector &ProxW(const Vector &x, const Vector &Weight, Vector *result) const;

		/*Action of the generalized Jacobian of the ProxW along the direction eta.
         Default: result = eta, i.e., lambbda = 0*/
		virtual Vector &CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const;

		/*Check the correctness of the Riemannian gradient and Riemannian Hessian.
		See details in the user manual*/
		virtual void CheckGradHessian(Variable x) const;
        
        /*Compute the minimim and maximum eigenvalues of the Riemannian Hessian at x*/
        virtual Vector MinMaxEigValHess(Variable x) const;
        
		/*Set the domain of the cost function.*/
		virtual void SetDomain(Manifold *inDomain);

		/*Obtain the domain manifold of the cost function*/
		inline Manifold *GetDomain(void) const { return Domain; };

		/*Mark whether the gradient is used in the algorithm or not.*/
		virtual void SetUseGrad(bool usegrad) const;

		/*Mark whether the action of the Hessian is used in the algorithm or not.*/
		virtual void SetUseHess(bool usehess) const;
        
        /*Mark whether the numerical gradient and numerical Hessian are used.*/
        virtual void SetNumGradHess(bool inNumGradHess) const;
        
		/*Get whether the gradient is used*/
		inline bool GetUseGrad(void) const { return UseGrad; };

		/*Get whether the action of the Hessian is used*/
		inline bool GetUseHess(void) const { return UseHess; };
        
        /*Get whether the numerical gradient and numerical Hessian are used*/
        inline bool GetNumGradHess(void) const { return NumGradHess; };
        
        mutable bool burnin;
        
	protected:
		Manifold *Domain; /*Domain of the cost function. It is required to be assigned.*/
		mutable bool UseGrad; /*Mark whether the gradient is used*/
		mutable bool UseHess; /*Mark whether the action of the Hessian is used.*/
        mutable bool NumGradHess;
        mutable integer N; /* for stochastic gradient type methods, the number of data, i.e., the N in \min_x \sum_{i = 1}^N f_i(x) */
	};

}; /*end of ROPTLIB namespace*/

#endif /* end of PROBLEM_H */
