/*
This file defines the inexact accelerated Riemannian proximal gradient method with acceleration on manifold with or without preconditioner.


Solvers --> SolversNSM --> SolversNSMPGLS --> IARPG

---- WH
*/

#ifndef IARPG_H
#define IARPG_H

#include "Solvers/SolversNSMPGLS.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

	class IARPG : public SolversNSMPGLS{
	public:
		/*The contructor of ARPG method. It calls the function Solvers::Initialization.
		INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		IARPG(const Problem *prob, const Variable *initialx);

        ~IARPG();
        
		/*Call Solvers::SetProbX function and indicate RCG does not need action of Hessian.
		INPUT:	prob is the problem which defines the cost function, gradient and possible the action of Hessian
		and specifies the manifold of domain.
		initialx is the initial iterate.*/
		virtual void SetProbX(const Problem *prob, const Variable *initialx);

		/*Setting parameters (member variables) to be default values */
		virtual void SetDefaultParams();
        
        /*Check whether the parameters in RPG are legal or not.*/
        virtual void CheckParams(void);
        
        /*Run the proximal gradient algorithm with acceleration*/
        void Run(void);
        
        /*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
        This function is used to set the parameters by the mapping*/
        virtual void SetParams(PARAMSMAP params);
        
        /*default: 5*/
        integer SGIterGap;
	protected:
        /*Print information specific to an algorithm*/
        virtual void PrintInfo(void);

        /*Print last information in an algorithm*/
        virtual void PrintFinalInfo(void);
        
        /*The variable x1 and x2 declear in Solvers.h are used for safeguard*/
        /*The variable z1 and z2 are iterates*/
        Variable z1;
        Variable z2;
        realdp fz1;
        realdp fz2;
        Vector gfz1;
        Vector gfz2;
        /*The variable y1 and y2 are auxiliary iterates*/
        Variable y1;
        Variable y2;
        realdp fy1;
        realdp fy2;
        Vector gfy1;
        Vector gfy2;
        
        realdp s1;
        realdp s2;
        
        Vector initDy;
        
        integer numrestart;
	};
}; /*end of ROPTLIB namespace*/
#endif /* end of IARPG_H */
