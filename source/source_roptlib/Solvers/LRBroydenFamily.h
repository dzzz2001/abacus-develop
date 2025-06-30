/*
 This file defines LRBroydenFamily
 
 Solvers --> SolversSM --> SolversSMLS --> LRBroydenFamily
 
 written by Shuguang Zhang, modified by Wen Huang
 */

#ifndef LRBROYDENFAMILY_H
#define LRBROYDENFAMILY_H

#include <cstring>
#include "Solvers/SolversSMLS.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    class LRBroydenFamily : public SolversSMLS{
    public:
        /*The contructor of LRBroydenFamily method. It calls the function Solvers::Initialization.
         INPUT : prob is the problem which defines the cost function, gradient and possible the action of Hessian
         and specifies the manifold of domain.
         initialx is the initial iterate.
         insoln is the true solution. It is not required and only used for research*/
        LRBroydenFamily(const Problem *prob, const Variable *initialx);
        
        /*Destructor. Delete the arrays and vectors used in LRBroydenFamily, i.e., series S and Y, and series RHO*/
        virtual ~LRBroydenFamily(void);
        
        /*Check whether the parameters about LRBroydenFamily are legal or not.*/
        virtual void CheckParams(void);
        
        /*Run the algorithm. New memory for S, Y and RHO. Then call SolversLS::Run*/
        virtual void Run(void);
        
        /*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
         This function is used to set the parameters by the mapping*/
        virtual void SetParams(PARAMSMAP params);
        
        /*specify whether the cost function is convex or not.
         If yes, then the initial Hessian approximation is a scalar times identity, where the scalar is to
         measure the magnitude of eigenvalues, otherwise, it is identity.
         Default: false*/
        bool isconvex;
        
        /*The same as \epsilon in [LF01, (3.2)]
         [LF01]: D.-H. Li and M. Fukushima. On the global convergence of the BFGS method for nonconvex unconstrained optimization problems.
         SIAM Journal on Optimization, 11(4):1054?064, 2001
         Default: 10^{-4}*/
        realdp nu;
        
        /*The same as \alpha in [LF01, (3.2)]
         [LF01]: D.-H. Li and M. Fukushima. On the global convergence of the BFGS method for nonconvex unconstrained optimization problems.
         SIAM Journal on Optimization, 11(4):1054?064, 2001
         Default: 1*/
        realdp mu;
        
        /*the number of pairs of s and y in Limited-memory version of quasi-Newton methods;  The same as \ell in [HGA2018]
         [HGA2018]: Wen Huang, K. A. Gallivan, and P.-A. Absil. A Riemannian BFGS Method without Differentiated Retraction for Nonconvex Optimization Problems
         SIAM Journal on Optimization, 28(1):pp.470-495, 2018
         Default: 4*/
        integer LengthSY;
        
        /*Control the choice of phi strategy
         default 0: LRDavidon
                 1: historical Davidon without boundary restriction, easily failed
                 2: historical Davidon with boundary restriction
         other values: same as \delta in hybrid Davidon-BFGS method */
        realdp phiset;
        
        /*If LMrestart is true, then we discard all pairs of (s_k, y_k) in limited-memory quasi-Newton when its number reaches LengthSY;
         Otherwise, we only discard the oldest one and add a new one, therefore the number of pairs of (s_k, y_k) is nondecreasing.*/
        bool LMrestart;
        
        /* parameter for the hybird version of BFGS and Davidon
         Default: 1.5 */
        realdp delta;
        
    protected:
        
        /*Compute the search direction. See [Zhang2024]
            [Zhang2024] Shuguang Zhang, Riemannian Broyden Family of Limited-memory Quasi-Newton Methods, Florida State University, dissertation, 2024
         */
        virtual void GetSearchDir(void);
        
        /*update the pairs of s and y. Add the latest one and remove the oldest one if necessary.
         transport them to the tangent space of x2*/
        virtual void UpdateData(void);
        
        /*Print information specific to LRBroydenFamily*/
        virtual void PrintInfo(void);
        
        /*Compute result = H v in LRBroydenFamily*/
        virtual Vector &HvLRBroydenFamily(const Vector &v, Vector *result);
        
        /*initial Hessian approximation in limited-memory RBroyden Family methods. It is a scalar times identity.*/
        virtual realdp InitialHessian(realdp inpss, realdp inpsy, realdp inpyy);
        
        /*Update the Hessian approximation for LRBroydenFamily if necessary*/
        virtual void UpdateDataLRBroydenFamily(void);
        
        /*Call Solvers::SetProbX function and set up the temporary objects for LRBroydenFamily algorithm.
         INPUT:    prob is the problem which defines the cost function, gradient and possible the action of Hessian
         and specifies the manifold of domain.
         initialx is the initial iterate.*/
        virtual void SetProbX(const Problem *prob, const Variable *initialx);
        
        /*Setting parameters (member variables) to be default values */
        virtual void SetDefaultParams(void);
        
        bool isupdated; /*Mark whether the (inverse) Hessian approximation is updated*/
        realdp betay, inpsy, inpss, inpyy, phi;  /*betay: \|\xi\| / \|\mathcal{T}_{R_\xi} \xi\| in the locking condition;
                                             phi: the coefficient (1 - phi) BFGS + phi DFP in Broyden family method
                                             inpsy: g(s, y); inpss: g(s, s); inpyy: g(y, y); */
        Vector s, y;/*the s, y, and u of current step*/
        Vector Py; /*the preconditioned y.*/
        Vector *S, *Y; /*The stored pairs of s and y*/
        realdp *store_phi; /*Store the phi_i^{(k)} value*/
        /*Store if the update is SR1*/
        integer *store_sr;
        integer sr_value;
        realdp *SY, *SS, *YY; /*Store the inner products between curvature pairs */
        realdp gamma; /*gamma: g(s, y) / g(y, y) for LRBroydenFamily and gamma: g(y, y) / g(s, y) for RTRSR1*/
        integer Currentlength; /*The current length of array S, Y and RHO*/
        integer beginidx, tempBegin; /*The starting index of S, Y and RHO at current iteration*/
    };
}; /*end of ROPTLIB namespace*/
#endif // end of RBROYDENFAMILY_H
