/*
 This file defines the class for the problem min_{U \in Gr(r,dim)} 1/N \sum_{i=1}^{N} \| (Uw_i)_{\Omega_i} - (a_i)_{\Omega_i} \|_2^2, w_i = U_{\Omega_i}^{\dagger}(z_i)_{\Omega_i}
 
 Problem --> GrassMatCompletion
 
 by Shuguang Zhang, modified by Wen Huang
 */

#ifndef GRASSMATCOMPLETION_H
#define GRASSMATCOMPLETION_H

#include "Manifolds/Grassmann.h"
//#include "Manifolds/Euclidean/EucVariable.h"
#include "Problems/Problem.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    class GrassMatCompletion : public Problem{
    public:
        GrassMatCompletion(integer *inir, integer *injc, realdp *invals, integer innz, integer inN, integer inD, integer inr);
        virtual ~GrassMatCompletion(void);
        
        virtual realdp f(const Variable &x) const;
        virtual realdp Stof(const Variable &x, const Vector &batch_index) const;
        
        virtual integer GetDataSize(void) const { return N; };
        virtual Vector &EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;
//        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;
        
        integer *ir;
        integer *jc;
        realdp *vals;
        mutable integer nz;
//        mutable integer N;
        mutable integer d;
        mutable integer r;
    
    protected:
        Vector &Getai(const Variable &x, const integer column, Vector *result) const;
        Vector &GetGradi(const Variable &x, const integer column, Vector *result) const;
    };
}; /*end of ROPTLIB namespace*/
#endif // end of GRASSMATCOMPLETION_H
