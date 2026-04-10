#include "phi_operator_gpu.h"
#include "phi_operator_kernel.cuh"
#include "dgemm_vbatch.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "source_base/module_device/device_check.h"

namespace ModuleGint
{

// Map a problem-size key (nw or max(nw1,nw2)) to one of three kernel buckets:
//   0 = small  (key <= 8)
//   1 = medium (8 < key <= 16)
//   2 = large  (key > 16)
static inline int gemm_bucket_of(int key)
{
    if (key <= 8) { return 0; }
    if (key <= 16) { return 1; }
    return 2;
}

template<typename Real>
PhiOperatorGpu<Real>::PhiOperatorGpu(std::shared_ptr<const GintGpuVars> gint_gpu_vars, cudaStream_t stream)
:gint_gpu_vars_(gint_gpu_vars), stream_(stream),
mgrids_num_(BatchBigGrid::get_bgrid_info()->get_mgrids_num()),
atoms_num_info_(BatchBigGrid::get_max_batch_size(), stream_, true),
bgrids_phi_len_(BatchBigGrid::get_max_batch_size(), stream_, true),
bgrids_phi_start_(BatchBigGrid::get_max_batch_size(), stream_, true),
atoms_iat_(BatchBigGrid::get_max_atoms_num(), stream_, true),
atoms_bgrids_rcoords_(BatchBigGrid::get_max_atoms_num(), stream_, true),
atoms_phi_start_(BatchBigGrid::get_max_atoms_num(), stream_, true),
mgrids_local_idx_batch_(BatchBigGrid::get_max_batch_size() 
    * BatchBigGrid::get_bgrid_info()->get_mgrids_num(), stream_, true),
gemm_m_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_n_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_k_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_lda_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_ldb_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_ldc_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_A_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_B_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_C_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true),
gemm_alpha_(BatchBigGrid::get_max_atom_pairs_num(), stream_, true)
{
    CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

template<typename Real>
PhiOperatorGpu<Real>::~PhiOperatorGpu()
{
    CHECK_CUDA(cudaEventDestroy(event_));
}

template<typename Real>
void PhiOperatorGpu<Real>::set_bgrid_batch(std::shared_ptr<BatchBigGrid> bgrid_batch)
{
    bgrid_batch_ = bgrid_batch;
    auto atoms_num_info_h = atoms_num_info_.get_host_ptr();
    auto bgrids_phi_len_h = bgrids_phi_len_.get_host_ptr();
    auto bgrids_phi_start_h = bgrids_phi_start_.get_host_ptr();
    auto atoms_iat_h = atoms_iat_.get_host_ptr();
    auto atoms_bgrids_rcoords_h = atoms_bgrids_rcoords_.get_host_ptr();
    auto atoms_phi_start_h = atoms_phi_start_.get_host_ptr();
    auto mgrids_local_idx_batch_h = mgrids_local_idx_batch_.get_host_ptr();
    int i = 0;
    int j = 0;
    int atoms_accum = 0;
    phi_len_ = 0;
    int phi_start = 0;
    std::vector<int> mgrids_local_idx;
    CHECK_CUDA(cudaEventSynchronize(event_));
    for (const auto& bgrid : bgrid_batch->get_bgrids())
    {
        atoms_num_info_h[i] = make_int2(bgrid->get_atoms_num(), atoms_accum);
        atoms_accum += bgrid->get_atoms_num();
        bgrids_phi_start_h[i] = phi_start;
        bgrid->set_mgrids_local_idx(mgrids_local_idx);
        std::copy(mgrids_local_idx.begin(), mgrids_local_idx.end(),
            mgrids_local_idx_batch_h + i * mgrids_num_);
        int phi_len_bgrid = 0;
        for (const auto& atom : bgrid->get_atoms())
        {
            atoms_iat_h[j] = atom->get_iat();
            Vec3d rcoord = bgrid->get_bgrid_atom_rcoord(atom);
            atoms_bgrids_rcoords_h[j] = make_double3(rcoord.x, rcoord.y, rcoord.z);
            atoms_phi_start_h[j] = phi_len_ + phi_len_bgrid;
            phi_len_bgrid += atom->get_nw();
            j++;
        }
        bgrids_phi_len_h[i] = phi_len_bgrid;
        phi_len_ += phi_len_bgrid * bgrid->get_mgrids_num();
        phi_start += phi_len_bgrid * bgrid->get_mgrids_num();
        i++;
    }

    atoms_num_info_.copy_host_to_device_async(bgrid_batch->get_batch_size());
    bgrids_phi_len_.copy_host_to_device_async(bgrid_batch->get_batch_size());
    bgrids_phi_start_.copy_host_to_device_async(bgrid_batch->get_batch_size());
    atoms_iat_.copy_host_to_device_async(bgrid_batch->get_atoms_num());
    atoms_bgrids_rcoords_.copy_host_to_device_async(bgrid_batch->get_atoms_num());
    atoms_phi_start_.copy_host_to_device_async(bgrid_batch->get_atoms_num());
    mgrids_local_idx_batch_.copy_host_to_device_async(bgrid_batch->get_batch_size() * mgrids_num_);
    CHECK_CUDA(cudaEventRecord(event_, stream_));
}

template<typename Real>
void PhiOperatorGpu<Real>::set_phi(Real* phi_d) const
{
    dim3 grid_dim(mgrids_num_, bgrid_batch_->get_batch_size());
    dim3 threads_per_block(64);
    set_phi_kernel<Real><<<grid_dim, threads_per_block, 0, stream_>>>(
        gint_gpu_vars_->nwmax,
        mgrids_num_,
        gint_gpu_vars_->nr_max,
        gint_gpu_vars_->dr_uniform,
        gint_gpu_vars_->ucell_atom_nwl_d,
        gint_gpu_vars_->atom_iw2_new_d,
        gint_gpu_vars_->atom_iw2_ylm_d,
        gint_gpu_vars_->atom_nw_d,
        gint_gpu_vars_->iat2it_d,
        gint_gpu_vars_->rcut_d,
        gint_gpu_vars_->psi_u_d,
        gint_gpu_vars_->dpsi_u_d,
        gint_gpu_vars_->mgrids_pos_d,
        atoms_iat_.get_device_ptr(),
        atoms_bgrids_rcoords_.get_device_ptr(),
        atoms_num_info_.get_device_ptr(),
        atoms_phi_start_.get_device_ptr(),
        bgrids_phi_len_.get_device_ptr(),
        phi_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::set_phi_dphi(double* phi_d, double* dphi_x_d, double* dphi_y_d, double* dphi_z_d) const
{
    dim3 grid_dim(mgrids_num_, bgrid_batch_->get_batch_size());
    dim3 threads_per_block(64);
    set_phi_dphi_kernel<<<grid_dim, threads_per_block, 0, stream_>>>(
        gint_gpu_vars_->nwmax,
        mgrids_num_,
        gint_gpu_vars_->nr_max,
        gint_gpu_vars_->dr_uniform,
        gint_gpu_vars_->ucell_atom_nwl_d,
        gint_gpu_vars_->atom_iw2_new_d,
        gint_gpu_vars_->atom_iw2_ylm_d,
        gint_gpu_vars_->atom_iw2_l_d,
        gint_gpu_vars_->atom_nw_d,
        gint_gpu_vars_->iat2it_d,
        gint_gpu_vars_->rcut_d,
        gint_gpu_vars_->psi_u_d,
        gint_gpu_vars_->dpsi_u_d,
        gint_gpu_vars_->mgrids_pos_d,
        atoms_iat_.get_device_ptr(),
        atoms_bgrids_rcoords_.get_device_ptr(),
        atoms_num_info_.get_device_ptr(),
        atoms_phi_start_.get_device_ptr(),
        bgrids_phi_len_.get_device_ptr(),
        phi_d,
        dphi_x_d,
        dphi_y_d,
        dphi_z_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::set_ddphi(double* ddphi_xx_d, double* ddphi_xy_d, double* ddphi_xz_d,
                               double* ddphi_yy_d, double* ddphi_yz_d, double* ddphi_zz_d) const
{
    // Since the underlying implementation of `set_ddphi` uses `ddphi +=` instead of `ddphi =`,
    // the ddphi array needs to be zeroed out at the beginning of the function.
    CHECK_CUDA(cudaMemsetAsync(ddphi_xx_d, 0, phi_len_ * sizeof(double), stream_));
    CHECK_CUDA(cudaMemsetAsync(ddphi_xy_d, 0, phi_len_ * sizeof(double), stream_));
    CHECK_CUDA(cudaMemsetAsync(ddphi_xz_d, 0, phi_len_ * sizeof(double), stream_));
    CHECK_CUDA(cudaMemsetAsync(ddphi_yy_d, 0, phi_len_ * sizeof(double), stream_));
    CHECK_CUDA(cudaMemsetAsync(ddphi_yz_d, 0, phi_len_ * sizeof(double), stream_));
    CHECK_CUDA(cudaMemsetAsync(ddphi_zz_d, 0, phi_len_ * sizeof(double), stream_));
    dim3 grid_dim(mgrids_num_, bgrid_batch_->get_batch_size());
    dim3 threads_per_block(64);
    set_ddphi_kernel<<<grid_dim, threads_per_block, 0, stream_>>>(
        gint_gpu_vars_->nwmax,
        mgrids_num_,
        gint_gpu_vars_->nr_max,
        gint_gpu_vars_->dr_uniform,
        gint_gpu_vars_->ucell_atom_nwl_d,
        gint_gpu_vars_->atom_iw2_new_d,
        gint_gpu_vars_->atom_iw2_ylm_d,
        gint_gpu_vars_->atom_iw2_l_d,
        gint_gpu_vars_->atom_nw_d,
        gint_gpu_vars_->iat2it_d,
        gint_gpu_vars_->rcut_d,
        gint_gpu_vars_->psi_u_d,
        gint_gpu_vars_->dpsi_u_d,
        gint_gpu_vars_->mgrids_pos_d,
        atoms_iat_.get_device_ptr(),
        atoms_bgrids_rcoords_.get_device_ptr(),
        atoms_num_info_.get_device_ptr(),
        atoms_phi_start_.get_device_ptr(),
        bgrids_phi_len_.get_device_ptr(),
        ddphi_xx_d,
        ddphi_xy_d,
        ddphi_xz_d,
        ddphi_yy_d,
        ddphi_yz_d,
        ddphi_zz_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_mul_vldr3(
    const Real* vl_d,
    const Real dr3,
    const Real* phi_d,
    Real* result_d) const
{
    dim3 grid_dim(mgrids_num_, bgrid_batch_->get_batch_size());
    dim3 threads_per_block(64);
    phi_mul_vldr3_kernel<Real><<<grid_dim, threads_per_block, 0, stream_>>>(
        vl_d,
        dr3,
        phi_d,
        mgrids_num_,
        mgrids_local_idx_batch_.get_device_ptr(),
        bgrids_phi_len_.get_device_ptr(),
        bgrids_phi_start_.get_device_ptr(),
        result_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_mul_phi(
    const Real* phi_d,
    const Real* phi_vldr3_d,
    HContainer<Real>& hRGint,
    Real* hr_d) const
{
    // Why bucket the atom pairs before calling gemm_tn_vbatch:
    // the vbatch dispatcher picks a single kernel template (a fixed
    // BLK_M x BLK_N x BLK_K tile) per launch, sized from the batch-wide
    // max(nw1, nw2). A real ABACUS batch mixes light atoms (nw ~ 4) with
    // transition metals (nw ~ 25) in the same launch, so that one template
    // is forced to fit the largest item and runs every small item on an
    // over-sized, register-heavy block. Splitting the batch into three
    // size tiers (see gemm_bucket_of) and issuing one vbatch launch per
    // tier lets each launch pick a template matched to its own size range.

    int ap_num = 0;
    int bucket_off[3] = {0, 0, 0};
    int bucket_cnt[3] = {0, 0, 0};
    int bmax_m[3] = {0, 0, 0};
    int bmax_n[3] = {0, 0, 0};

    auto* h_A   = gemm_A_.get_host_ptr();
    auto* h_B   = gemm_B_.get_host_ptr();
    auto* h_C   = gemm_C_.get_host_ptr();
    auto* h_lda = gemm_lda_.get_host_ptr();
    auto* h_ldb = gemm_ldb_.get_host_ptr();
    auto* h_ldc = gemm_ldc_.get_host_ptr();
    auto* h_m   = gemm_m_.get_host_ptr();
    auto* h_n   = gemm_n_.get_host_ptr();
    auto* h_k   = gemm_k_.get_host_ptr();

    const auto* atoms_num_h  = atoms_num_info_.get_host_ptr();
    const auto* phi_start_h  = atoms_phi_start_.get_host_ptr();
    const auto& bgrids       = bgrid_batch_->get_bgrids();
    const int batch_size     = bgrid_batch_->get_batch_size();

    CHECK_CUDA(cudaEventSynchronize(event_));

    for (int b = 0; b < 3; b++)
    {
        bucket_off[b] = ap_num;
        for (int i = 0; i < batch_size; i++)
        {
            const auto& bgrid = bgrids[i];
            const int phi_len_mgrid = bgrid->get_phi_len();
            const int mgrids_num = bgrid->get_mgrids_num();
            const int pre_atoms = atoms_num_h[i].y;
            const int atoms_num = bgrid->get_atoms_num();
            const auto& atoms = bgrid->get_atoms();
            for (int ia_1 = 0; ia_1 < atoms_num; ia_1++)
            {
                const auto& atom_1 = atoms[ia_1];
                const int iat_1 = atom_1->get_iat();
                const int nw1 = atom_1->get_nw();
                const int phi_1_offset = phi_start_h[pre_atoms + ia_1];
                const auto& r_1 = atom_1->get_R();

                for (int ia_2 = 0; ia_2 < atoms_num; ia_2++)
                {
                    const auto& atom_2 = atoms[ia_2];
                    const int iat_2 = atom_2->get_iat();
                    if (iat_1 > iat_2) { continue; }

                    const int nw2 = atom_2->get_nw();
                    // TN dispatch key is max(nw1, nw2) -- both feed kernel M/N.
                    if (gemm_bucket_of(std::max(nw1, nw2)) != b) { continue; }

                    const int hr_offset = hRGint.find_matrix_offset(
                        iat_1, iat_2, r_1 - atom_2->get_R());
                    if (hr_offset == -1) { continue; }

                    const int phi_2_offset = phi_start_h[pre_atoms + ia_2];

                    h_A[ap_num] = phi_d + phi_1_offset;
                    h_B[ap_num] = phi_vldr3_d + phi_2_offset;
                    h_C[ap_num] = hr_d + hr_offset;
                    h_lda[ap_num] = phi_len_mgrid;
                    h_ldb[ap_num] = phi_len_mgrid;
                    h_ldc[ap_num] = nw2;
                    h_m[ap_num] = nw1;
                    h_n[ap_num] = nw2;
                    h_k[ap_num] = mgrids_num;

                    bmax_m[b] = std::max(bmax_m[b], nw1);
                    bmax_n[b] = std::max(bmax_n[b], nw2);
                    ap_num++;
                }
            }
        }
        bucket_cnt[b] = ap_num - bucket_off[b];
    }

    gemm_A_.copy_host_to_device_async(ap_num);
    gemm_B_.copy_host_to_device_async(ap_num);
    gemm_C_.copy_host_to_device_async(ap_num);
    gemm_lda_.copy_host_to_device_async(ap_num);
    gemm_ldb_.copy_host_to_device_async(ap_num);
    gemm_ldc_.copy_host_to_device_async(ap_num);
    gemm_m_.copy_host_to_device_async(ap_num);
    gemm_n_.copy_host_to_device_async(ap_num);
    gemm_k_.copy_host_to_device_async(ap_num);
    CHECK_CUDA(cudaEventRecord(event_, stream_));

    for (int b = 0; b < 3; b++)
    {
        if (bucket_cnt[b] == 0) { continue; }
        const int off = bucket_off[b];
        gemm_tn_vbatch<Real>(bmax_m[b],
                        bmax_n[b],
                        mgrids_num_,
                        gemm_m_.get_device_ptr() + off,
                        gemm_n_.get_device_ptr() + off,
                        gemm_k_.get_device_ptr() + off,
                        gemm_A_.get_device_ptr() + off,
                        gemm_lda_.get_device_ptr() + off,
                        gemm_B_.get_device_ptr() + off,
                        gemm_ldb_.get_device_ptr() + off,
                        gemm_C_.get_device_ptr() + off,
                        gemm_ldc_.get_device_ptr() + off,
                        bucket_cnt[b],
                        stream_,
                        nullptr);
    }
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_mul_dm(
    const Real* phi_d,
    const Real* dm_d,
    const HContainer<Real>& dm,
    const bool is_symm,
    Real* phi_dm_d)
{
    CHECK_CUDA(cudaMemsetAsync(phi_dm_d, 0, phi_len_ * sizeof(Real), stream_));

    int ap_num = 0;
    int bucket_off[3] = {0, 0, 0};
    int bucket_cnt[3] = {0, 0, 0};
    int bmax_n[3] = {0, 0, 0};
    int bmax_k[3] = {0, 0, 0};

    auto* h_A     = gemm_A_.get_host_ptr();
    auto* h_B     = gemm_B_.get_host_ptr();
    auto* h_C     = gemm_C_.get_host_ptr();
    auto* h_lda   = gemm_lda_.get_host_ptr();
    auto* h_ldb   = gemm_ldb_.get_host_ptr();
    auto* h_ldc   = gemm_ldc_.get_host_ptr();
    auto* h_m     = gemm_m_.get_host_ptr();
    auto* h_n     = gemm_n_.get_host_ptr();
    auto* h_k     = gemm_k_.get_host_ptr();
    auto* h_alpha = gemm_alpha_.get_host_ptr();

    const auto* atoms_num_h = atoms_num_info_.get_host_ptr();
    const auto* phi_start_h = atoms_phi_start_.get_host_ptr();
    const auto& bgrids      = bgrid_batch_->get_bgrids();
    const int batch_size    = bgrid_batch_->get_batch_size();

    CHECK_CUDA(cudaEventSynchronize(event_));

    for (int b = 0; b < 3; b++)
    {
        bucket_off[b] = ap_num;
        for (int i = 0; i < batch_size; i++)
        {
            const auto& bgrid = bgrids[i];
            const int phi_len_mgrid = bgrid->get_phi_len();
            const int pre_atoms = atoms_num_h[i].y;
            const int atoms_num = bgrid->get_atoms_num();
            const auto& atoms = bgrid->get_atoms();
            for (int ia_1 = 0; ia_1 < atoms_num; ia_1++)
            {
                const auto& atom_1 = atoms[ia_1];
                const int iat_1 = atom_1->get_iat();
                const int nw1 = atom_1->get_nw();
                const int phi_1_offset = phi_start_h[pre_atoms + ia_1];
                const auto& r_1 = atom_1->get_R();

                const int ia_2_start = is_symm ? ia_1 : 0;
                for (int ia_2 = ia_2_start; ia_2 < atoms_num; ia_2++)
                {
                    const auto& atom_2 = atoms[ia_2];
                    const int nw2 = atom_2->get_nw();

                    // NN dispatch key is nw2 (gemm N dim; M = mgrids_num_).
                    if (gemm_bucket_of(nw2) != b) { continue; }

                    const int iat_2 = atom_2->get_iat();
                    const int dm_offset = dm.find_matrix_offset(
                        iat_1, iat_2, r_1 - atom_2->get_R());
                    if (dm_offset == -1) { continue; }

                    const int phi_dm_offset = phi_start_h[pre_atoms + ia_2];

                    h_A[ap_num] = phi_d + phi_1_offset;
                    h_B[ap_num] = dm_d + dm_offset;
                    h_C[ap_num] = phi_dm_d + phi_dm_offset;
                    h_lda[ap_num] = phi_len_mgrid;
                    h_ldb[ap_num] = nw2;
                    h_ldc[ap_num] = phi_len_mgrid;
                    h_m[ap_num] = mgrids_num_;
                    h_n[ap_num] = nw2;
                    h_k[ap_num] = nw1;
                    if (is_symm)
                    {
                        h_alpha[ap_num] = ia_1 == ia_2 ? Real(1.0) : Real(2.0);
                    }

                    bmax_n[b] = std::max(bmax_n[b], nw2);
                    bmax_k[b] = std::max(bmax_k[b], nw1);
                    ap_num++;
                }
            }
        }
        bucket_cnt[b] = ap_num - bucket_off[b];
    }

    gemm_A_.copy_host_to_device_async(ap_num);
    gemm_B_.copy_host_to_device_async(ap_num);
    gemm_C_.copy_host_to_device_async(ap_num);
    gemm_lda_.copy_host_to_device_async(ap_num);
    gemm_ldb_.copy_host_to_device_async(ap_num);
    gemm_ldc_.copy_host_to_device_async(ap_num);
    gemm_m_.copy_host_to_device_async(ap_num);
    gemm_n_.copy_host_to_device_async(ap_num);
    gemm_k_.copy_host_to_device_async(ap_num);
    if (is_symm)
    {
        // if is_symm == false, gemm_alpha_ is always 1.0 and is skipped on device
        gemm_alpha_.copy_host_to_device_async(ap_num);
    }
    CHECK_CUDA(cudaEventRecord(event_, stream_));

    for (int b = 0; b < 3; b++)
    {
        if (bucket_cnt[b] == 0) { continue; }
        const int off = bucket_off[b];
        auto alpha_ptr = is_symm ? (gemm_alpha_.get_device_ptr() + off) : nullptr;
        gemm_nn_vbatch<Real>(mgrids_num_,
                        bmax_n[b],
                        bmax_k[b],
                        gemm_m_.get_device_ptr() + off,
                        gemm_n_.get_device_ptr() + off,
                        gemm_k_.get_device_ptr() + off,
                        gemm_A_.get_device_ptr() + off,
                        gemm_lda_.get_device_ptr() + off,
                        gemm_B_.get_device_ptr() + off,
                        gemm_ldb_.get_device_ptr() + off,
                        gemm_C_.get_device_ptr() + off,
                        gemm_ldc_.get_device_ptr() + off,
                        bucket_cnt[b],
                        stream_,
                        alpha_ptr);
    }
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_dot_phi(
    const Real* phi_i_d,
    const Real* phi_j_d,
    Real* rho_d) const
{
    dim3 grid_dim(mgrids_num_, bgrid_batch_->get_batch_size());
    dim3 threads_per_block(64);
    phi_dot_phi_kernel<Real><<<grid_dim, threads_per_block, sizeof(Real) * 32, stream_>>>(
        phi_i_d,
        phi_j_d,
        mgrids_num_,
        mgrids_local_idx_batch_.get_device_ptr(),
        bgrids_phi_len_.get_device_ptr(),
        bgrids_phi_start_.get_device_ptr(),
        rho_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_dot_dphi(
    const double* phi_d,
    const double* dphi_x_d,
    const double* dphi_y_d,
    const double* dphi_z_d,
    double* fvl_d) const
{
    dim3 grid_dim(bgrid_batch_->get_max_atoms_num_per_bgrid(),
                  bgrid_batch_->get_batch_size());
    dim3 threads_per_block(32);
    phi_dot_dphi_kernel<<<grid_dim, threads_per_block, sizeof(double) * 32 * 3, stream_>>>(
        phi_d,
        dphi_x_d,
        dphi_y_d,
        dphi_z_d,
        mgrids_num_,
        bgrids_phi_len_.get_device_ptr(),
        atoms_num_info_.get_device_ptr(),
        atoms_phi_start_.get_device_ptr(),
        atoms_iat_.get_device_ptr(),
        gint_gpu_vars_->iat2it_d,
        gint_gpu_vars_->atom_nw_d,
        fvl_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

template<typename Real>
void PhiOperatorGpu<Real>::phi_dot_dphi_r(
    const double* phi_d,
    const double* dphi_x_d,
    const double* dphi_y_d,
    const double* dphi_z_d,
    double* svl_d) const
{
    dim3 grid_dim(mgrids_num_,
                  bgrid_batch_->get_batch_size());
    dim3 threads_per_block(32);
    phi_dot_dphi_r_kernel<<<grid_dim, threads_per_block, sizeof(double) * 32 * 6, stream_>>>(
        phi_d,
        dphi_x_d,
        dphi_y_d,
        dphi_z_d,
        mgrids_num_,
        bgrids_phi_len_.get_device_ptr(),
        atoms_num_info_.get_device_ptr(),
        atoms_phi_start_.get_device_ptr(),
        atoms_iat_.get_device_ptr(),
        atoms_bgrids_rcoords_.get_device_ptr(),
        gint_gpu_vars_->mgrids_pos_d,
        gint_gpu_vars_->iat2it_d,
        gint_gpu_vars_->atom_nw_d,
        svl_d);
    CHECK_LAST_CUDA_ERROR("kernel launch");
}

// Explicit instantiations
template class PhiOperatorGpu<double>;
template class PhiOperatorGpu<float>;

}