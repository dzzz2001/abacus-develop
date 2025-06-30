/*
 This file defines the class for the problem min_{U \in Gr(p,n)} -1/N \sum_{i=1}^{N} x_i^T U U^T x_i
 
 Problem --> GrassPCA
 
 by Shuguang Zhang, modified by Wen Huang
 */

#ifndef GRASSPCA_H
#define GRASSPCA_H

#include "Manifolds/Grassmann.h"
//#include "Manifolds/Euclidean/EucVariable.h"
#include "Problems/Problem.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    class GrassPCA : public Problem{
    public:
        GrassPCA(Vector inA, integer inN, integer inn, integer inp);
        virtual ~GrassPCA(void);
        
        virtual realdp f(const Variable &x) const;
        virtual realdp Stof(const Variable &x, const Vector &batch_index) const;
        
        virtual integer GetDataSize(void) const { return N; };
        virtual Vector &EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;
        
        //virtual realdp fit(const Variable &x, const int it) const;
        //virtual Vector &EucGradit(const Variable &x, const int it, Vector *result) const;
        
//        virtual integer GetDataSize(void) const { return n; };
        virtual Vector &EucGrad(const Variable &x, Vector *result) const;
        virtual Vector &EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const;
        
        Vector A;
        integer N;
        integer n;
        integer p;
    };
}; /*end of ROPTLIB namespace*/
#endif // end of EUCQUADRATIC_H
