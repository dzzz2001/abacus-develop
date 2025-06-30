#ifndef ESOLVER_RDMFT_PW_H
#define ESOLVER_RDMFT_PW_H

#include "esolver_fp.h"

// #include "module_optimizer/"
namespace ModuleESolver
{

  template <typename T, typename Device = base_device::DEVICE_CPU>
class ESolver_RDMFT_PW : public ESolver_FP
{
  public:
    ESolver_RDMFT_PW();
    ~ESolver_RDMFT_PW();

    void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    void after_all_runners(UnitCell& ucell) override;

    void others(UnitCell& ucell, const int istep) override;

    void runner(UnitCell& ucell, const int istep) override;


    protected:
    // virtual void before_scf(UnitCell& ucell, const int istep) override;

    // virtual void iter_init(UnitCell& ucell, const int istep, const int iter) override;

    // virtual void hamilt2density_single(UnitCell& ucell, const int istep, const int iter, const double ethr) override;

    // virtual void update_pot(UnitCell& ucell, const int istep, const int iter) override;

    // virtual void iter_finish(UnitCell& ucell, const int istep, int& iter) override;

    // virtual void after_scf(UnitCell& ucell, const int istep) override;

    // virtual void others(UnitCell& ucell, const int istep) override;

};

}

#endif // ESOLVER_RDMFT_PW_H