#include "esolver_rdmft_pw.h"

namespace ModuleESolver
{

template <typename TK,  typename Device>
 ESolver_RDMFT_PW<TK, Device>::ESolver_RDMFT_PW()
{
    // Constructor implementation
}


template <typename TK,  typename Device>
 ESolver_RDMFT_PW<TK, Device>::~ESolver_RDMFT_PW()
{
    //  Destructor implementation
}

template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    // Implementation of before_all_runners
}

template <typename TK,  typename Device>
double ESolver_RDMFT_PW<TK, Device>::cal_energy()
{
    // Implementation of cal_energy
    return 0.0;
}

template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    // Implementation of cal_force
}

template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    // Implementation of cal_stress
}

template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::after_all_runners(UnitCell& ucell)
{
    // Implementation of after_all_runners
}

template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::others(UnitCell& ucell, const int istep)
{
    // Implementation of others
}
  
template <typename TK,  typename Device>
void ESolver_RDMFT_PW<TK, Device>::runner(UnitCell& ucell, const int istep)
{
    // Implementation of runner
}


template class ESolver_RDMFT_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_RDMFT_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_RDMFT_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_RDMFT_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif

}