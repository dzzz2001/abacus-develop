#include "interp.cuh"
#include "gint_rho.cuh"
#include "sph.cuh"

namespace GintKernel
{
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
                        double* psi)
{
    const int bcell_id = blockIdx.x;
    const int num_atoms = atoms_per_bcell[bcell_id];
    const int bcell_start = start_idx_per_bcell[bcell_id];
    const int mcell_id = blockIdx.y;
    const double mcell_pos_x = mcell_pos[mcell_id];
    const double mcell_pos_y = mcell_pos[bxyz + mcell_id];
    const double mcell_pos_z = mcell_pos[2 * bxyz + mcell_id];
    for(int atom_id = threadIdx.x; atom_id < num_atoms; atom_id+=blockDim.x)
    {
        const double dr_x = dr_x_part[bcell_start + atom_id] + mcell_pos_x;
        const double dr_y = dr_y_part[bcell_start + atom_id] + mcell_pos_y;
        const double dr_z = dr_z_part[bcell_start + atom_id] + mcell_pos_z;
        double dist = sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z);
        const int atype = __ldg(atom_type + bcell_start + atom_id);
        const int nwl = __ldg(ucell_atom_nwl + atype);
        if(dist < rcut[atype])
        {
            if (dist < 1.0E-9)
            {
                dist += 1.0E-9;
            }
            double dr[3] = {dr_x / dist, dr_y / dist, dr_z / dist};
            double ylma[49];
            spherical_harmonics(dr, nwl, ylma, ylmcoef);
            int psi_idx = (bcell_id * bxyz + mcell_id) * max_atom * nwmax
                                + atom_id * nwmax;
            interpolate(dist,
                        delta_r,
                        atype,
                        nwmax,
                        nr_max,
                        atom_nw,
                        atom_iw2_new,
                        psi_u,
                        ylma,
                        atom_iw2_ylm,
                        psi,
                        psi_idx,
                        1);
        }
    }
}

__global__ void psir_dot(const int bxyz,
                         const int vec_size,
                         const double* __restrict__ vec_a_g,
                         const double* __restrict__  vec_b_g,
                         double** results_g)
{
    extern __shared__ double s_data[];
    const int tid = threadIdx.x;
    const int offset = blockIdx.x * bxyz * vec_size + blockIdx.y * vec_size;
    const double* vec_a_mcell = vec_a_g + offset;
    const double* vec_b_mcell = vec_b_g + offset;

    s_data[tid] = 0.0;

    for(unsigned int k = tid; k < vec_size; k += blockDim.x)
    {
        s_data[tid] += vec_a_mcell[k] * vec_b_mcell[k];
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *results_g[blockIdx.x*bxyz + blockIdx.y] = s_data[0];
    }
}
} // namespace GintKernel