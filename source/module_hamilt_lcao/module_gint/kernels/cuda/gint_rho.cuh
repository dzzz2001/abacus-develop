#ifndef GINT_RHO_CUH
#define GINT_RHO_CUH

#include <cuda_runtime.h>
#include <stdint.h>
namespace GintKernel
{

/**
 * @brief CUDA kernel to calculate psir.
 *
 * This kernel calculates the wave function psi using the provided input
 * parameters.
 */
__global__ void get_psi(const double* const ylmcoef,
                        const double delta_r,
                        const int bxyz,
                        const int nwmax,
                        const int max_atom,
                        const int* const ucell_atom_nwl,
                        const bool* const atom_iw2_new,
                        const int* const atom_iw2_ylm,
                        const int* const atom_nw,
                        const double* const rcut,
                        const int nr_max,
                        const double* const psi_u,
                        const double* const mcell_pos,
                        const double* const dr_x_part,
                        const double* const dr_y_part,
                        const double* const dr_z_part,
                        const int* const atoms_per_bcell,
                        const uint8_t* const atom_type,
                        const int* const start_idx_per_bcell,
                        bool* mat_cal_flag,
                        double* psi);

__global__ void psir_dot(const int bxyz,
                         const int vec_size,
                         const double* __restrict__ vec_a_g,
                         const double* __restrict__  vec_b_g,
                         double** results_g);

} // namespace GintKernel
#endif // GINT_RHO_CUH