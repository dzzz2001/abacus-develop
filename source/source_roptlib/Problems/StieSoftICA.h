/*
This file defines the class for the problem (details in (12.1.1) in Wen Huang's Thesis)
\min_{X \in \St(p, n)} - \Sum_{i=1}^N \| diag(X^T C_i X^T) \|_F^2,
where St(p, n) is the Stiefel manifold, C_i are given symmetric matricies.

Problem --> StieSoftICA

---- WH
*/

#ifndef STIESOFTICA_H
#define STIESOFTICA_H

#include "Manifolds/Stiefel.h"
#include "Problems/Problem.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class StieSoftICA : public Problem{
	public:
		StieSoftICA(Vector inCs, integer inp);
        
        virtual realdp f(const Variable &x) const;
        virtual realdp Stof(const Variable &x, const Vector &batch_index) const;
        
        virtual integer GetDataSize(void) const { return N; };
        virtual Vector &EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;
        
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;
        
        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;

		Vector Cs;
		mutable integer n;
		mutable integer p;
		mutable integer N;
	};
}; /*end of ROPTLIB namespace*/
#endif // end of STIESOFTICA_H
