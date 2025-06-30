/*
This file defines the class for a sparse variable PCA model on the Grassmann manifold
min_X - tr(X^T B^T B X) + \lambda \|X\|_{2, 1}, where B is a m-by-n matrix, and X \in Gr(p, n),
 and \|X\|_{2, 1} = \sum_{i = 1}^n \|X_{i, :}\|_2.
Typically, n > m > p.

Problem --> GrassSVPCA

---- WH
*/

#ifndef GRASSSVPCA_H
#define GRASSSVPCA_H

#include "Manifolds/Grassmann.h"
#include "Problems/Problem.h"
#include "Solvers/IRPG.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class GrassSVPCA : public Problem{
	public:
		GrassSVPCA(Vector inB, realdp inlambda, integer inn, integer inm, integer inp);
		virtual ~GrassSVPCA();
		virtual realdp f(const Variable &x) const;
        virtual realdp g(const Variable &x) const;
        
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;
        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;
        
        virtual Vector &ProxW(const Vector &x, const Vector &Weight, Vector *result) const;
        virtual Vector &CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const;
        
        /*The create the weight matrix*/
        virtual Vector &PreConditioner(const Variable &x, const Vector &eta, Vector *result) const;

		Vector B;
        realdp lambda;
		mutable integer n;
        mutable integer m;
		mutable integer p;
        realdp L;
	};
}; /*end of ROPTLIB namespace*/
#endif
