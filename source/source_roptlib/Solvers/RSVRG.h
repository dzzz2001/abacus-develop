//
//  RSVRG.cpp
//  Riemannian stochastic variance-reduction gradient method
//
//    Solvers --> SolversSM --> SolversSMSto --> SolversSMSVRG -->  RSVRG
//
//  by Shuguang Zhang, modified by Wen Huang
//

#ifndef RSVRG_H
#define RSVRG_H

#include "Solvers/SolversSMSVRG.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    class RSVRG : public SolversSMSVRG{
    public:
        /*The contructor of RSD method. It calls the function Solvers::Initialization.
         INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
         and specifies the manifold of domain.
         initialx is the initial iterate.*/
        RSVRG(const Problem *prob, const Variable *initialx);
        
        /*Check whether the parameters about linesearch algorithms are legal or not.*/
        virtual void CheckParams(void);
        
    protected:
        /*Set the search direction to be negative gradient*/
        virtual void GetSearchDir(void);
        
        /*Call Solvers::SetProbX function and indicate RCG does not need action of Hessian.
         INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
         and specifies the manifold of domain.
         initialx is the initial iterate.*/
        virtual void SetProbX(const Problem *prob, const Variable *initialx);
        
        /*Setting parameters (member variables) to be default values */
        virtual void SetDefaultParams(void);
    };
}; /*end of ROPTLIB namespace*/
#endif // end of RSVRG_H
