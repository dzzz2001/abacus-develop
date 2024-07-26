#include "kernels/cuda/cuda_tools.cuh"
#include "gint_rho_gpu.h"
#include "module_base/ylm.h"
#include "module_hamilt_lcao/module_gint/gint_tools.h"
#include "omp.h"
namespace GintKernel
{

void gtask_rho(const Grid_Technique& gridt,
               const int grid_index_ij,
               const UnitCell& ucell,
               double* dr_part,
               uint8_t* atoms_type,
               int* atoms_num_info,
               int& atoms_per_z)         
{
    atoms_per_z = 0;
    for (int z_index = 0; z_index < gridt.nbzp; z_index++)
    {
        int grid_index = grid_index_ij + z_index;
        int bcell_start_index = gridt.bcell_start[grid_index];
        int na_grid = gridt.how_many_atoms[grid_index];
        atoms_num_info[2 * z_index] = na_grid;
        atoms_num_info[2 * z_index + 1] = atoms_per_z;
        for (int id = 0; id < na_grid; id++)
        {
            int mcell_index = bcell_start_index + id;
            int imcell = gridt.which_bigcell[mcell_index];
            int iat = gridt.which_atom[mcell_index];
            int it_temp = ucell.iat2it[iat];

            dr_part[atoms_per_z * 3] = gridt.meshball_positions[imcell][0]
                      - gridt.tau_in_bigcell[iat][0];
            dr_part[atoms_per_z * 3 + 1] = gridt.meshball_positions[imcell][1]
                      - gridt.tau_in_bigcell[iat][1];
            dr_part[atoms_per_z * 3 + 2] = gridt.meshball_positions[imcell][2]
                      - gridt.tau_in_bigcell[iat][2];
            atoms_type[atoms_per_z] = it_temp;
            atoms_per_z++;
        }
    }
}

void alloc_mult_dot_rho(const Grid_Technique& gridt,
                        const UnitCell& ucell,
                        const int grid_index_ij,
                        const int max_atom,
                        const int lgd,
                        const int nczp,
                        const int* atoms_num_info,
                        double* const psir_ylm_g,
                        double* const psir_dm_g,
                        double* const dm_matrix_g,
                        double* mat_alpha,
                        int* mat_m,
                        int* mat_n,
                        int* mat_k,
                        int* mat_lda,
                        int* mat_ldb,
                        int* mat_ldc,
                        double** mat_A,
                        double** mat_B,
                        double** mat_C,
                        int& max_m,
                        int& max_n,
                        int& atom_pair_num,
                        double* rho_g,
                        double** dot_product)
{
    int tid = 0;
    int dot_count = 0;
    max_m = 0;
    max_n = 0;
    const int nwmax=ucell.nwmax;
    // generate matrix multiplication tasks
    for (int z_index = 0; z_index < gridt.nbzp; z_index++)
    {
        int grid_index = grid_index_ij + z_index;
        int bcell_start_index = gridt.bcell_start[grid_index];
        int bcell_start_psir = atoms_num_info[2 * z_index + 1] * gridt.bxyz * nwmax;
        int na_grid = atoms_num_info[2 * z_index];

        for (int atom1 = 0; atom1 < gridt.how_many_atoms[grid_index]; atom1++)
        {
            int mcell_index1 = bcell_start_index + atom1;
            int iat1 = gridt.which_atom[mcell_index1];
            int it1 = ucell.iat2it[iat1];
            int lo1
                = gridt.trace_lo[ucell.itiaiw2iwt(it1, ucell.iat2ia[iat1], 0)];
            int nw1 = ucell.atoms[it1].nw;

            for (int atom2 = atom1; atom2 < gridt.how_many_atoms[grid_index];
                 atom2++)
            {
                int mcell_index2 = bcell_start_index + atom2;
                int iat2 = gridt.which_atom[mcell_index2];
                int it2 = ucell.iat2it[iat2];
                int lo2 = gridt.trace_lo[ucell.itiaiw2iwt(it2,
                                                          ucell.iat2ia[iat2],
                                                          0)];
                int nw2 = ucell.atoms[it2].nw;

                int mat_A_idx = bcell_start_psir + atom2 * nwmax;
                int mat_B_idx = lgd * lo1 + lo2;
                int mat_C_idx = bcell_start_psir + atom1 * nwmax;

                mat_alpha[tid] = atom2 == atom1 ? 1 : 2;
                mat_m[tid] = gridt.bxyz;
                mat_n[tid] = nw1;
                mat_k[tid] = nw2;
                mat_lda[tid] = nwmax * na_grid;
                mat_ldb[tid] = lgd;
                mat_ldc[tid] = nwmax * na_grid;
                mat_A[tid] = psir_ylm_g + mat_A_idx;
                mat_B[tid] = dm_matrix_g + mat_B_idx;
                mat_C[tid] = psir_dm_g + mat_C_idx;

                if (mat_m[tid] > max_m)
                {
                    max_m = mat_m[tid];
                }

                if (mat_n[tid] > max_n)
                {
                    max_n = mat_n[tid];
                }

                tid++;
            }
        }

        // generate vec dot product tasks
        int* vindex = Gint_Tools::get_vindex(gridt.bxyz,
                                             gridt.bx,
                                             gridt.by,
                                             gridt.bz,
                                             nczp,
                                             gridt.start_ind[grid_index],
                                             gridt.ncy * nczp);
        for (int i = 0; i < gridt.bxyz; i++)
        {
            dot_product[dot_count] = rho_g + vindex[i];
            dot_count++;
        }
        
        delete[] vindex;
    }
    atom_pair_num = tid;
}

void alloc_mult_dot_rho_k(const Grid_Technique& gridt,
                        const UnitCell& ucell,
                        const hamilt::HContainer<double>* dm,
                        const int grid_index_ij,
                        const int max_atom,
                        const int lgd,
                        const int nczp,
                        const int* atoms_num_info,
                        double* const psir_ylm_g,
                        double* const psir_dm_g,
                        double* const dm_matrix_g,
                        double* mat_alpha,
                        int* mat_m,
                        int* mat_n,
                        int* mat_k,
                        int* mat_lda,
                        int* mat_ldb,
                        int* mat_ldc,
                        double** mat_A,
                        double** mat_B,
                        double** mat_C,
                        int& max_m,
                        int& max_n,
                        int& atom_pair_num,
                        double* dmr_g,
                        double* rho_g,
                        double** dot_product,
                        std::vector<bool> is_malloced,
                        cudaStream_t stream)
{
    int tid = 0;
    int dot_count = 0;
    max_m = 0;
    max_n = 0;
    const int nwmax=ucell.nwmax;
    // generate matrix multiplication tasks
    for (int z_index = 0; z_index < gridt.nbzp; z_index++)
    {
        const int grid_index = grid_index_ij + z_index;
        const int bcell_start_index = gridt.bcell_start[grid_index];
        const int bcell_start_psir = atoms_num_info[2 * z_index + 1] * gridt.bxyz * nwmax;
        const int na_grid = atoms_num_info[2 * z_index];

        for (int atom1 = 0; atom1 < gridt.how_many_atoms[grid_index]; atom1++)
        {
            const int mcell_index1 = bcell_start_index + atom1;
            const int iat1 = gridt.which_atom[mcell_index1];
            const int it1 = ucell.iat2it[iat1];
            const int lo1
                = gridt.trace_lo[ucell.itiaiw2iwt(it1, ucell.iat2ia[iat1], 0)];
            const int id1 = gridt.which_unitcell[bcell_start_index + atom1];
            const int nw1 = ucell.atoms[it1].nw;
            const int DM_start = gridt.nlocstartg[iat1];
            const int r1x = gridt.ucell_index2x[id1];
            const int r1y = gridt.ucell_index2y[id1];
            const int r1z = gridt.ucell_index2z[id1];

            for (int atom2 = atom1; atom2 < gridt.how_many_atoms[grid_index];
                 atom2++)
            {
                const int mcell_index2 = bcell_start_index + atom2;
                const int iat2 = gridt.which_atom[mcell_index2];
                const int it2 = ucell.iat2it[iat2];
                const int id2 = gridt.which_unitcell[bcell_start_index + atom2];
                const int lo2 = gridt.trace_lo[ucell.itiaiw2iwt(it2,
                                                          ucell.iat2ia[iat2],
                                                          0)];
                const int nw2 = ucell.atoms[it2].nw;
                const int offset = gridt.find_offset(id1, id2, iat1, iat2);
                if(offset == -1)
                {
                    continue;
                }
                const int iatw = DM_start + gridt.find_R2st[iat1][offset];
                const int r2x = gridt.ucell_index2x[id2];
                const int r2y = gridt.ucell_index2y[id2];
                const int r2z = gridt.ucell_index2z[id2];
                const int drx = r1x - r2x;
                const int dry = r1y - r2y;
                const int drz = r1z - r2z;
                double *tmp_mat = dm->find_matrix(iat1, iat2, drx, dry, drz)->get_pointer();
                bool need_transfer = false; 
#pragma omp critical
                {
                    if(is_malloced[iatw] == false)
                    {
                        is_malloced[iatw] = true;
                        need_transfer = true;
                    }
                }
                if(need_transfer){
                    checkCuda(cudaMemcpy(
                        dmr_g + iatw,tmp_mat,
                        sizeof(double) * nw1 * nw2,
                        cudaMemcpyHostToDevice));
                }
                const int mat_A_idx = bcell_start_psir + atom2 * nwmax;
                const int mat_C_idx = bcell_start_psir + atom1 * nwmax;

                mat_alpha[tid] = atom2 == atom1 ? 1 : 2;
                mat_m[tid] = gridt.bxyz;
                mat_n[tid] = nw1;
                mat_k[tid] = nw2;
                mat_lda[tid] = nwmax * na_grid;
                mat_ldb[tid] = nw1;
                mat_ldc[tid] = nwmax * na_grid;
                mat_A[tid] = psir_ylm_g + mat_A_idx;
                mat_B[tid] = dmr_g + iatw;
                mat_C[tid] = psir_dm_g + mat_C_idx;

                if (mat_m[tid] > max_m)
                {
                    max_m = mat_m[tid];
                }

                if (mat_n[tid] > max_n)
                {
                    max_n = mat_n[tid];
                }

                tid++;
            }
        }

        // generate vec dot product tasks
        int* vindex = Gint_Tools::get_vindex(gridt.bxyz,
                                             gridt.bx,
                                             gridt.by,
                                             gridt.bz,
                                             nczp,
                                             gridt.start_ind[grid_index],
                                             gridt.ncy * nczp);
        for (int i = 0; i < gridt.bxyz; i++)
        {
            dot_product[dot_count] = rho_g + vindex[i];
            dot_count++;
        }
        
        delete[] vindex;
    }
    atom_pair_num = tid;
}

} // namespace GintKernel