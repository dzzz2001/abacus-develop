/*
This file defines the class for a sparse PCA model on the Oblique manifold
see paper: Weakly correlated sparse components with nearly orthonormal loadings
min_X  \|X^T A^T A X - D^2\|_F^2 + lambda \|X\|_1, where A is a m-by-n matrix, and X \in Oblique(n, p).
Typically, n > m > p.

Problem --> ObliqueSPCA

---- WH
*/

#ifndef OBLIQUESPCA_H
#define OBLIQUESPCA_H

#include "Manifolds/Stiefel.h"
#include "Problems/Problem.h"
#include "Solvers/IRPG.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class ObliqueSPCA : public Problem{
	public:
		ObliqueSPCA(Vector inA, realdp inlambda, integer inn, integer inm, integer inp);
		virtual ~ObliqueSPCA();
		virtual realdp f(const Variable &x) const;
        
        virtual realdp g(const Variable &x) const;
        
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;
        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;
        
        virtual Vector &ProxW(const Vector &x, const Vector &Weight, Vector *result) const;
        virtual Vector &CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const;
        
        /*The create the weight matrix*/
        virtual Vector &PreConditioner(const Variable &x, const Vector &eta, Vector *result) const;

		Vector A;
        Vector Dsq;
        realdp lambda;
		mutable integer n;
        mutable integer m;
		mutable integer p;
        realdp L;
	};
}; /*end of ROPTLIB namespace*/
#endif
