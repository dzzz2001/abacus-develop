/*
This file defines the class for the problem
min_{X \in R_r^{m by n}} \|P_Omega (invH(X) - A) \|_F^2 + lambda \|X\|_1,
where R_r^{m by n} is the set of m by n matrices with rank r, A is a given m by n matrix,
Omega is an index set, and invH() denotes the 2-d inverse Haar wavelet transformation or
the 2-d DCT transformation

Problem --> FRankETextureInpainting.h

---- WH
*/

#ifndef FRANKETEXTUREINPAINTING_H
#define FRANKETEXTUREINPAINTING_H

#include "Problems/Problem.h"
#include "Others/def.h"
#include "Manifolds/FixedRankE.h"

#ifdef ROPTLIB_WITH_FFTW

/*Define the namespace*/
namespace ROPTLIB{

	class FRankETextureInpainting : public Problem{
	public:
        /* inir, injc, injcc are used to denotes the index of Omega. inA denotes the given matrix. inlambda is the lambda in the objective function.
        inm, inn, and inr denote the size and the rank of the matrix X. inlengthW is used for the preconditioner. type is used to denote the transformation invH(),
        i.e., type = 1: Haar, type = 2: DCT*/
		FRankETextureInpainting(unsigned long *inir, unsigned long *injc, unsigned long *injcc, unsigned long innzmax, Vector inA, realdp inlambda, integer inm, integer inn, integer inr, integer inlengthW, integer intype);
		virtual ~FRankETextureInpainting();
		virtual realdp f(const Variable &x) const;
        virtual realdp g(const Variable &x) const;

		virtual Vector &EucGrad(const Variable &x, Vector *result) const;
		virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;
        
        virtual Vector &ProxW(const Vector &x, const Vector &Weight, Vector *result) const;
        virtual Vector &CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const;
        
        /*The create the weight matrix*/
        virtual Vector &PreConditioner(const Variable &x, const Vector &eta, Vector *result) const;

        unsigned long *ir;
        unsigned long *jc;
        unsigned long *jcc;
        unsigned long nzmax;
        integer type;
		Vector A;
        realdp lambda;
		integer m;
		integer n;
		integer r;
        integer lengthW;
        realdp L;
	};
}; /*end of ROPTLIB namespace*/

#endif //FRANKETEXTUREINPAINTING_H
#endif

