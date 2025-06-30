/*
This file defines the class for the problem
min_{X \in R_r^{m by n}} 0.5 \|P_{\Omega}(X) - P_{\Omega}(A)\|_F^2, 
where R_r{m by n} is the set of m by n matrices with rank r, \Omega is an index set, A_{ij}, (i, j) \in \Omega
are given.

Problem --> FRankE3FMatCompletion

---- WH
*/

#ifndef FRANKE3FMATCOMPLETION_H
#define FRANKE3FMATCOMPLETION_H

#include "Problems/Problem.h"
#include "Others/def.h"
#include "Manifolds/Element.h"
#include "Manifolds/FixedRankE3F.h"

/*Define the namespace*/
namespace ROPTLIB{

	class FRankE3FMatCompletion : public Problem{
	public:
		FRankE3FMatCompletion(unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdp *inV, integer innz, integer inm, integer inn, integer inr);

		virtual ~FRankE3FMatCompletion();

		/*0.5 \|P_omaga(X) - P_omega(A)\|_F^2*/
		virtual realdp f(const Variable &x) const;

		/*P_omaga(X) - P_omega(A)*/
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;

		/*P_omaga(etax)*/
        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;

		/*compute result = P_Omega(U D V^T), where the indices of Omega are stored in inir and injc.*/
        static void ProjecOmegaUDVT(const realdp *U, const realdp *D, const realdp *V, integer inm, integer inn, integer inr, unsigned long *inir, unsigned long *injc, unsigned long *injcc, integer nz, realdp *result);

		unsigned long *ir;
		unsigned long *jc;
        unsigned long *jcc;
		realdp *vals;
		mutable integer nz;
		mutable integer m;
		mutable integer n;
		mutable integer r;
        
//        mutable SparseMatrix *SMEucRepGrad;
//        mutable SparseMatrix *SMEucRepHess;
	};
}; /*end of ROPTLIB namespace*/
#endif
