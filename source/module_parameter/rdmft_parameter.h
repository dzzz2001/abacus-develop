#ifndef RDMFT_PARA_H
#define RDMFT_PARA_H

#include <string>
#include <vector>

/**
 * @brief input parameters used in reduced density matrix functional theory (RDMFT)
 *
 */
struct RDMFT_para
{
    std::string    rdmft_fnal= "power"; // functional type: muller, power, hf, pnof etc.

    // default to use two loops, if set to 1, then do simultaneous optimization
    int loop_layer = 2;
    
    int max_iter_orb = 10; // max iteration for orbital optimization
    int max_iter_occ = 10; // max iteration for occupation optimization

   std::string  occ_opt_method="sd"; // method for occupation optimization
   std::string  orb_opt_method="cg"; // method for orbital optimization

    // line search algorithm for optimization
    // armijo: Armijo
    // wolfe: classical Wolfe condition
    // swolfe: Strong wolfe condition
    // exact: exact condition
   std::string ls_algo = "wolfe"; 

    // conjugate gradient algorithm 
    // FR: FLETCHER_REEVES
    // PR: POLAK_RIBIERE
    // DY: DAI_YUAN
    // HZ: HAGER_ZHANG
    // HS: HESTENES_STIEFEL
   std::string ls_cg_algo = "fr";


    // initial step type  
    // BB: 
    // QuadInt:
    std::string initstep_type; 
 
    double ls_gtol; // line search gradient tolerance
    double ls_ftol; // line search function value tolerance

};

#endif // RDMFT_PARA_H