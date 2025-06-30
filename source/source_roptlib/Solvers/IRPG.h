/*
This file defines the inexact Riemannian proximal gradient method on manifold with or without preconditioner.

Solvers --> SolversNSM --> SolversNSMPGLS --> IRPG

---- WH
*/

#ifndef IRPG_H
#define IRPG_H

#include "Solvers/SolversNSMPGLS.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class IRPG : public SolversNSMPGLS{
	public:
		/*The contructor of RPG method. It calls the function Solvers::Initialization.
		INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		IRPG(const Problem *prob, const Variable *initialx);

        ~IRPG();
        
		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams();
        
        /*Run the proximal gradient algorithm.*/
        void Run(void);
        
	protected:
        realdp alphaBB;
        
        /*Print information specific to an algorithm*/
        virtual void PrintInfo(void);

        /*Print last information in an algorithm*/
        virtual void PrintFinalInfo(void);
	};
}; /*end of ROPTLIB namespace*/
#endif /* end of IRPG_H */
