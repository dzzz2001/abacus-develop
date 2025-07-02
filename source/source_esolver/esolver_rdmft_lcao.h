#ifndef ESOLVER_RDMFT_LCAO_H
#define ESOLVER_RDMFT_LCAO_H

#include "esolver_ks_lcao.h"

// #include "module_optimizer/problem.h"

namespace ModuleESolver
{

template <typename TK, typename TR>
class ESolver_RDMFT_LCAO : public ESolver_KS_LCAO<TK, TR>
{
  public:
    ESolver_RDMFT_LCAO();
    ~ESolver_RDMFT_LCAO();

    void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    void after_all_runners(UnitCell& ucell) override;

    void others(UnitCell& ucell, const int istep) override;

    void runner(UnitCell& ucell, const int istep) override;


    protected:

    virtual void initialize(UnitCell& ucell, const int istep);

    virtual void before_scf(UnitCell& ucell, const int istep) override;

    // virtual void iter_init(UnitCell& ucell, const int istep, const int iter) override;

    // virtual void hamilt2density_single(UnitCell& ucell, const int istep, const int iter, const double ethr) override;

    // virtual void update_pot(UnitCell& ucell, const int istep, const int iter) override;

    // virtual void iter_finish(UnitCell& ucell, const int istep, int& iter) override;

    // virtual void after_scf(UnitCell& ucell, const int istep) override;

    // virtual void others(UnitCell& ucell, const int istep) override;

    private:
    // // void set_default_parameters(); // line search parameters
    // void initialize_line_search(); // initialize line search parameters
    // void get_search_direction(); // 
    // void perform_line_search(); //
    // void update_data();
    // void print_line_search_info(); // print line search information
    
    // void zoom();

    void occupation_optimization();
    void orbital_optimization();

    void joint_optimization();

    void initialize_density_matrix(); // initialize the  natural orbitals, occupation numbers
    // void initialize_eta(); // initialize the occupation numbers


    std::string occupation_optimization_method = "sd";
    std::string orbital_optimization_method = "cg";

    int nkpt; // number of k-points
    int nbands; // number of local bands
    int nbasis; // number of basis functions
    
  
};

}

#endif // ESOLVER_RDMFT_LCAO_H