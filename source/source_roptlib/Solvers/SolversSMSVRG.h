/*
 This is the base class for stochastic gradient method and its variances.
 
 Stochastic quasi-Newton idea in Hiroyuki Kasai, Hiroyuki Sato, and Bamdev Mishra. Riemannian stochastic quasi-Newton algorithm with variance reduction and its convergence analysis. In International Conference on Artificial Intelligence and Statistics, pages 269–278. PMLR, 2018.
 
 Solvers --> SolversSM --> SolversSMSto --> SolversSMSVRG
 
 Written by Shuguang Zhang, modified by Wen Huang
 */

#ifndef SOLVERSSMSVRG_H
#define SOLVERSSMSVRG_H

//#include <cmath>
#include <iostream>
#include <list>
#include <ctime>
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include "Solvers/SolversSMSto.h"
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    class SolversSMSVRG : public SolversSMSto{
    public:
        /*Run the algorithm. This function gives the framework for linesearch based methods*/
        virtual void Run(void);
        
        /*Check whether the parameters about linesearch algorithms are legal or not.*/
        virtual void CheckParams(void);
        
        /*PARAMSMAP is defined in "def.h" and it is a map from string to realdp, i.e., std::map<std::string, realdp> .
         This function is used to set the parameters by the mapping*/
        virtual void SetParams(PARAMSMAP params);
        
        /* ===============public parameters below================= */
        
//        /*fixed learning rate
//         Default: 0.0001 */
//        realdp Initstepsize;
        
//        /*the coefficient of decaying learning rate
//         Default: 0.999*/
//        realdp Decaying_rate;
        
//        /*the minimum stepsize allowed in the linesearch algorithm
//         Default: machine eps*/
//        realdp Minstepsize;
//        
//        /*the maximum stepsize allowed in the linesearch algorithm
//         Default: 1000 */
//        realdp Maxstepsize;
        
//        /*Initial stepsize at the first iteration
//         Default: 1*/
//        realdp Initstepsize;
        
//        /*When the iterate is close to the minimizer (see the annotation of Accuracy), then fixed
//         the stepsize to the Finalstepsize if Finalstepsize > 0. If Finalstepsize <= 0, then the proposed
//         initial stepsize is used as the accepted stepsize.
//         Default: 1*/
//        realdp Finalstepsize;
        
        /* Frequncy Value, default: 10 */
        integer FreqM;
        
//        /* batch size, default: 1 */
//        integer BatchSize;

    protected:
        
        /*Delete objects that are used in this class*/
        virtual ~SolversSMSVRG(void);
        
        /*Compute the search direction. It is a pure virtual function.*/
        virtual void GetSearchDir(void) = 0; // required to be overload in derived class if the derived class is not abstract
        
//        virtual void InitialStepSize(void);
        
//        /*Print information in every few iterations specific to an algorithm*/
//        virtual void PrintInfo(void);
//        
//        /*Print last information in an algorithm*/
//        virtual void PrintFinalInfo(void);
        
        /*Setting parameters (member variables) to be default values */
        virtual void SetDefaultParams(void);
        
//        /*Evaluate the cost function h(stepsize) = f(R_{x_1}(stepsize * eta1))*/
//        virtual realdp h(void);
//        
//        /*Evaluate the derivative of cost function h, i.e., h'(stepsize) = \frac{d}{d stepsize} f(R_{x_1}(stepsize * eta1))*/
//        virtual realdp dh(void);
        
        /*When one iteration, some algorithms need to update some information. For example,
         quasi-Newton methods need to update the Hessian approximation and nonlinear conjugate gradient
         needs to update the search direction. They are done in the following function*/
        virtual void UpdateData(void);
        
        /*Generate \tilde{M} and ifSR for Hessian approximation in LS-SBroyden*/
        virtual void HessianPrep(void);
        
        // parameters
        realdp initiallength;    /*The initial stepsize at an iteration*/
//        realdp stepsize;        /*The step size*/
//        integer nsf;            /*the number of batch function evaluations*/
//        integer nsg;            /*the number of batch gradient evaluations*/
    };
}; /*end of ROPTLIB namespace*/
#endif // end of SOLVERSSMSVRG_H
