#include "esolver_rdmft_lcao.h"

#include "source_hsolver/hsolver.h" // for hsolver::set_diagethr_ks


#include "Problems/Problem.h"
#include "Manifolds/Manifold.h"

namespace ModuleESolver
{

template <typename TK, typename TR>
 ESolver_RDMFT_LCAO<TK, TR>::ESolver_RDMFT_LCAO()
{
    this->classname = "ESolver_RDMFT_LCAO";
}


template <typename TK, typename TR>
ESolver_RDMFT_LCAO<TK, TR>::~ESolver_RDMFT_LCAO()
{
    //  Destructor implementation
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    // Implementation of before_all_runners, adapted from esolver_ks.cpp
   ESolver_KS_LCAO<TK, TR>::before_all_runners(ucell, inp);

    // There are a few choices for different psi initialization methods.
    // 1) Use the psi from KS solver after a few iterations
    if (PARAM.inp.rdmftp.rdmft_init_method == "ks")
    {
        ESolver_KS_LCAO<TK, TR>::runner(ucell, 0);
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_RDMFT_LCAO", "Unknown rdmft_init_method: " 
            + PARAM.inp.rdmftp.rdmft_init_method);
    }

   nkpt = this->psi -> get_nk();
   nbands = this->psi -> get_nbands();
   nbasis = this->psi -> get_nbasis();

}

template <typename TK, typename TR>

double ESolver_RDMFT_LCAO<TK, TR>::cal_energy()
{
    // Implementation of cal_energy
    return 0.0;
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    // Implementation of cal_force
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    // Implementation of cal_stress
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::after_all_runners(UnitCell& ucell)
{
    // Implementation of after_all_runners
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::others(UnitCell& ucell, const int istep)
{
    // Implementation of others
}
  
template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "runner");
    ModuleBase::timer::tick(this->classname, "runner");

    // ESolver_KS_LCAO<TK, TR>::runner(ucell, istep);

    //----------------------------------------------------------------
    // 1) before_scf (electronic iteration loops)
    //----------------------------------------------------------------
    this->before_scf(ucell, istep);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT RDMFT ");

    // 2) SCF iterations
    bool conv_esolver = false;
    this->niter = this->maxniter;
    this->diag_ethr = PARAM.inp.pw_diag_thr;

    int loop_layer = PARAM.inp.rdmftp.loop_layer;
    int niter_occ = 0;
    int niter_orb = 0;

    for(int iter = 0; iter < this->maxniter; iter++)
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "RDMFT LOOP ITER", iter);

        // choose to either use double loop or single loop
        if( loop_layer == 2)
        {
            occupation_optimization();
            orbital_optimization();
            // ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "<ITER OCC>", iter_occ);
            // Implementation of double loop
            // Call the necessary functions for double loop

            // ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "RDMFT Iteration Index: OCC, ORB", iter_occ, iter_orb);
            // Implementation of the inner loop
            // Call the necessary functions for the inner loop
        
            // niter_occ 
            // check convergence
            if( conv_esolver )
            {
                this->niter = iter;
                break;
            }
        }
        else
        {
            ModuleBase::WARNING_QUIT("ESolver_RDMFT_LCAO", "loop_layer  1 has not been implemented yet");
        }
    }


    ModuleBase::timer::tick(this->classname, "runner");
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::initialize(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "initialize");
    ModuleBase::timer::tick(this->classname, "initialize");

    // set the optimization of occupation to use steepest descent
    occupation_optimization_method = PARAM.inp.rdmftp.occ_opt_method;
    
    // set the optimization of orbital to use conjugate gradient
    orbital_optimization_method = PARAM.inp.rdmftp.orb_opt_method;

    // Implementation of initialize
    ModuleBase::timer::tick(this->classname, "initialize");
}


template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "before_scf");
    ModuleBase::timer::tick(this->classname, "before_scf");

    ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);
    // Implementation of before_scf
    ModuleBase::timer::tick(this->classname, "before_scf");
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::occupation_optimization()
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "occupation_optimization");
    ModuleBase::timer::tick(this->classname, "occupation_optimization");

    for(int iter_occ = 0; iter_occ < PARAM.inp.rdmftp.max_iter_occ; iter_occ++)
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "<ITER OCC>", iter_occ);
    }

    ModuleBase::timer::tick(this->classname, "occupation_optimization");

}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::orbital_optimization()
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "orbital_optimization");
    ModuleBase::timer::tick(this->classname, "orbital_optimization");

    // Implementation of orbital_optimization
    for(int iter_orb = 0; iter_orb < PARAM.inp.rdmftp.max_iter_orb; iter_orb++)
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "<ITER ORB>", iter_orb);
    }

    ModuleBase::timer::tick(this->classname, "orbital_optimization");
    return;
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::joint_optimization()
{


    // Implementation of joint_optimization
}

template <typename TK, typename TR>
void ESolver_RDMFT_LCAO<TK, TR>::initialize_density_matrix()
{
    ModuleBase::TITLE("ESolver_RDMFT_LCAO", "initialize_density_matrix");
    ModuleBase::timer::tick(this->classname, "initialize_density_matrix");


    //    // Use the psi from KS solver
//         // the number of iterations is controlled still by the scf_nmax for KS solver.
//         int istep = 0;

//         // ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);

//         bool conv_esolver = false;
//         this->niter = this->maxniter;
//         this->diag_ethr = PARAM.inp.pw_diag_thr;

//         for(int iter = 1; iter < this->maxniter; iter++)
//         {
//             ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "RDMFT INIT LOOP ITER", iter);
//             // call the before_scf function to initialize the psi
//             // this->before_scf(ucell, 1);

//         ESolver_KS_LCAO<TK, TR>::iter_init(ucell, istep, iter);

//         std::cout << "before hamilt2rho_single and diag_ethr = " << this->diag_ethr << std::endl;
//         //----------------------------------------------------------------
//         // use Hamiltonian to obtain charge density
// 		//----------------------------------------------------------------
//         ESolver_KS_LCAO<TK, TR>::hamilt2rho(ucell, istep, iter, this->diag_ethr);
//             std::cout << "after hamilt2rho" << std::endl;
//         ESolver_KS_LCAO<TK, TR>::iter_finish(ucell, istep, iter, conv_esolver);

//             std::cout << "after iter_finish" << std::endl;

//             // check convergence
//             if (conv_esolver || this->oscillate_esolver)
//             {
//                 this->niter = iter;
//                 if (this->oscillate_esolver)
//                 {
//                     std::cout << " !! Density oscillation is found, STOP HERE !!" << std::endl;
//                 }
//                 break;
//             }

//         } // end scf iterations

//         ESolver_KS_LCAO<TK, TR>::after_scf(ucell, istep, conv_esolver);

        ModuleBase::timer::tick(this->classname, "initialize_density_matrix");
    }

// template <typename TK, typename TR>
// void ESolver_RDMFT_LCAO<TK, TR>::initialize_eta()
// {
//     ModuleBase::TITLE("ESolver_RDMFT_LCAO", "initialize_eta");
//     ModuleBase::timer::tick(this->classname, "initialize_eta"); 

//     // 1) Use the eta from KS solver after a few iterations
//     if (PARAM.inp.rdmftp.eta_init_method == "ks")
//     {
//         // Use the eta from KS solver
//         this->pelec->wg  = this->pelec->wg;
//     }
//     // Implementation of initialize_eta
//     ModuleBase::timer::tick(this->classname, "initialize_eta");
// }

template class ESolver_RDMFT_LCAO<double, double>;
template class ESolver_RDMFT_LCAO<std::complex<double>, double>;
template class ESolver_RDMFT_LCAO<std::complex<double>, std::complex<double>>;
}