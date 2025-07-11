#include <omp.h>
#include "hsolver_lcao.h"

#ifdef __MPI
#include "diago_scalapack.h"
#include "source_base/scalapack_connector.h"
#else
#include "diago_lapack.h"
#endif

#ifdef __CUSOLVERMP
#include "diago_cusolvermp.h"
#endif

#ifdef __ELPA
#include "diago_elpa.h"
#include "diago_elpa_native.h"
#endif

#ifdef __CUDA
#include "diago_cusolver.h"
#include <cuda_runtime.h>
#endif

#ifdef __PEXSI
#include "diago_pexsi.h"
#endif

#include "source_base/global_variable.h"
#include "source_estate/elecstate_tools.h"
#include "source_base/memory.h"
#include "source_base/timer.h"
#include "source_estate/elecstate_lcao.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_hsolver/parallel_k2d.h"
#include "module_parameter/parameter.h"

namespace hsolver
{

template <typename T, typename Device>
void HSolverLCAO<T, Device>::solve(hamilt::Hamilt<T>* pHamilt,
                                   psi::Psi<T>& psi,
                                   elecstate::ElecState* pes,
                                   const bool skip_charge)
{
    ModuleBase::TITLE("HSolverLCAO", "solve");
    ModuleBase::timer::tick("HSolverLCAO", "solve");

    if (this->method != "pexsi")
    {
        if (PARAM.globalv.kpar_lcao > 1
            && (this->method == "genelpa" || this->method == "elpa" || this->method == "scalapack_gvx"))
        {
#ifdef __MPI
            this->parakSolve(pHamilt, psi, pes, PARAM.globalv.kpar_lcao);
#endif
        }
        else if (PARAM.globalv.kpar_lcao > 1 && this->method == "cusolver")
        {
            this->parakSolve_cusolver(pHamilt, psi, pes);
        }
        else if (PARAM.globalv.kpar_lcao == 1)
        {
            /// Loop over k points for solve Hamiltonian to eigenpairs(eigenvalues and eigenvectors).
            for (int ik = 0; ik < psi.get_nk(); ++ik)
            {
                /// update H(k) for each k point
                pHamilt->updateHk(ik);

                /// find psi pointer for each k point
                psi.fix_k(ik);

                /// solve eigenvector and eigenvalue for H(k)
                this->hamiltSolvePsiK(pHamilt, psi, &(pes->ekb(ik, 0)));
            }
        }
        else
        {
            ModuleBase::WARNING_QUIT("HSolverLCAO::solve",
                                     "This method and KPAR setting is not supported for lcao basis in ABACUS!");
        }

        elecstate::calculate_weights(pes->ekb,
                                     pes->wg,
                                     pes->klist,
                                     pes->eferm,
                                     pes->f_en,
                                     pes->nelec_spin,
                                     pes->skip_weights);

        auto _pes_lcao = dynamic_cast<elecstate::ElecStateLCAO<T>*>(pes);
        elecstate::calEBand(_pes_lcao->ekb, _pes_lcao->wg, _pes_lcao->f_en);
        elecstate::cal_dm_psi(_pes_lcao->DM->get_paraV_pointer(), _pes_lcao->wg, psi, *(_pes_lcao->DM));
        _pes_lcao->DM->cal_DMR();

        if (!skip_charge)
        {
            // used in scf calculation
            // calculate charge by eigenpairs(eigenvalues and eigenvectors)
            pes->psiToRho(psi);
        }
        else
        {
            // used in nscf calculation
        }
    }
    else if (this->method == "pexsi")
    {
#ifdef __PEXSI // other purification methods should follow this routine
        DiagoPexsi<T> pe(ParaV);
        for (int ik = 0; ik < psi.get_nk(); ++ik)
        {
            /// update H(k) for each k point
            pHamilt->updateHk(ik);
            psi.fix_k(ik);
            // solve eigenvector and eigenvalue for H(k)
            pe.diag(pHamilt, psi, nullptr);
        }
        auto _pes = dynamic_cast<elecstate::ElecStateLCAO<T>*>(pes);
        pes->f_en.eband = pe.totalFreeEnergy;
        // maybe eferm could be dealt with in the future
        _pes->dmToRho(pe.DM, pe.EDM);
#endif
    }

    ModuleBase::timer::tick("HSolverLCAO", "solve");
    return;
}

template <typename T, typename Device>
void HSolverLCAO<T, Device>::hamiltSolvePsiK(hamilt::Hamilt<T>* hm, psi::Psi<T>& psi, double* eigenvalue)
{
    ModuleBase::TITLE("HSolverLCAO", "hamiltSolvePsiK");
    ModuleBase::timer::tick("HSolverLCAO", "hamiltSolvePsiK");

    if (this->method == "scalapack_gvx")
    {
#ifdef __MPI
        DiagoScalapack<T> sa;
        sa.diag(hm, psi, eigenvalue);
#endif
    }
#ifdef __ELPA
    else if (this->method == "genelpa")
    {
        DiagoElpa<T> el;
        el.diag(hm, psi, eigenvalue);
    }
    else if (this->method == "elpa")
    {
        DiagoElpaNative<T> el;
        el.diag(hm, psi, eigenvalue);
    }
#endif
#ifdef __CUDA
    else if (this->method == "cusolver")
    {
        DiagoCusolver<T> cs(this->ParaV);
        cs.diag(hm, psi, eigenvalue);
    }
#ifdef __CUSOLVERMP
    else if (this->method == "cusolvermp")
    {
        DiagoCusolverMP<T> cm;
        cm.diag(hm, psi, eigenvalue);
    }
#endif
#endif
#ifndef __MPI
    else if (this->method == "lapack") // only for single core
    {
        DiagoLapack<T> la;
        la.diag(hm, psi, eigenvalue);
    }
#endif
    else
    {
        ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "This method is not supported for lcao basis in ABACUS!");
    }

    ModuleBase::timer::tick("HSolverLCAO", "hamiltSolvePsiK");
}

template <typename T, typename Device>
void HSolverLCAO<T, Device>::parakSolve(hamilt::Hamilt<T>* pHamilt,
                                        psi::Psi<T>& psi,
                                        elecstate::ElecState* pes,
                                        int kpar)
{
#ifdef __MPI
    ModuleBase::timer::tick("HSolverLCAO", "parakSolve");
    auto k2d = Parallel_K2D<T>();
    k2d.set_kpar(kpar);
    int nbands = this->ParaV->get_nbands();
    int nks = psi.get_nk();
    int nrow = this->ParaV->get_global_row_size();
    int nb2d = this->ParaV->get_block_size();
    k2d.set_para_env(psi.get_nk(), nrow, nb2d, GlobalV::NPROC, GlobalV::MY_RANK, PARAM.inp.nspin);
    /// set psi_pool
    const int zero = 0;
    int ncol_bands_pool
        = numroc_(&(nbands), &(nb2d), &(k2d.get_p2D_pool()->coord[1]), &zero, &(k2d.get_p2D_pool()->dim1));
    /// Loop over k points for solve Hamiltonian to charge density
    for (int ik = 0; ik < k2d.get_pKpoints()->get_max_nks_pool(); ++ik)
    {
        // if nks is not equal to the number of k points in the pool
        std::vector<int> ik_kpar;
        int ik_avail = 0;
        for (int i = 0; i < k2d.get_kpar(); i++)
        {
            if (ik + k2d.get_pKpoints()->startk_pool[i] < nks && ik < k2d.get_pKpoints()->nks_pool[i])
            {
                ik_avail++;
            }
        }
        if (ik_avail == 0)
        {
            ModuleBase::WARNING_QUIT("HSolverLCAO::solve", "ik_avail is 0!");
        }
        else
        {
            ik_kpar.resize(ik_avail);
            for (int i = 0; i < ik_avail; i++)
            {
                ik_kpar[i] = ik + k2d.get_pKpoints()->startk_pool[i];
            }
        }
        k2d.distribute_hsk(pHamilt, ik_kpar, nrow);
        /// global index of k point
        int ik_global = ik + k2d.get_pKpoints()->startk_pool[k2d.get_my_pool()];
        auto psi_pool = psi::Psi<T>(1, ncol_bands_pool, k2d.get_p2D_pool()->nrow, k2d.get_p2D_pool()->nrow, true);
        ModuleBase::Memory::record("HSolverLCAO::psi_pool", nrow * ncol_bands_pool * sizeof(T));
        if (ik_global < psi.get_nk() && ik < k2d.get_pKpoints()->nks_pool[k2d.get_my_pool()])
        {
            /// local psi in pool
            psi_pool.fix_k(0);
            hamilt::MatrixBlock<T> hk_pool = hamilt::MatrixBlock<T>{k2d.hk_pool.data(),
                                                                    (size_t)k2d.get_p2D_pool()->get_row_size(),
                                                                    (size_t)k2d.get_p2D_pool()->get_col_size(),
                                                                    k2d.get_p2D_pool()->desc};
            hamilt::MatrixBlock<T> sk_pool = hamilt::MatrixBlock<T>{k2d.sk_pool.data(),
                                                                    (size_t)k2d.get_p2D_pool()->get_row_size(),
                                                                    (size_t)k2d.get_p2D_pool()->get_col_size(),
                                                                    k2d.get_p2D_pool()->desc};
            /// solve eigenvector and eigenvalue for H(k)
            if (this->method == "scalapack_gvx")
            {
                DiagoScalapack<T> sa;
                sa.diag_pool(hk_pool, sk_pool, psi_pool, &(pes->ekb(ik_global, 0)), k2d.POOL_WORLD_K2D);
            }
#ifdef __ELPA
            else if (this->method == "genelpa")
            {
                DiagoElpa<T> el;
                el.diag_pool(hk_pool, sk_pool, psi_pool, &(pes->ekb(ik_global, 0)), k2d.POOL_WORLD_K2D);
            }
            else if (this->method == "elpa")
            {
                DiagoElpaNative<T> el;
                el.diag_pool(hk_pool, sk_pool, psi_pool, &(pes->ekb(ik_global, 0)), k2d.POOL_WORLD_K2D);
            }
#endif
            else
            {
                ModuleBase::WARNING_QUIT("HSolverLCAO::solve",
                                         "This type of eigensolver for k-parallelism diagnolization is not supported!");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        ModuleBase::timer::tick("HSolverLCAO", "collect_psi");
        for (int ipool = 0; ipool < ik_kpar.size(); ++ipool)
        {
            int source = k2d.get_pKpoints()->get_startpro_pool(ipool);
            MPI_Bcast(&(pes->ekb(ik_kpar[ipool], 0)), nbands, MPI_DOUBLE, source, MPI_COMM_WORLD);
            int desc_pool[9];
            std::copy(k2d.get_p2D_pool()->desc, k2d.get_p2D_pool()->desc + 9, desc_pool);
            if (k2d.get_my_pool() != ipool)
            {
                desc_pool[1] = -1;
            }
            psi.fix_k(ik_kpar[ipool]);
            Cpxgemr2d(nrow,
                      nbands,
                      psi_pool.get_pointer(),
                      1,
                      1,
                      desc_pool,
                      psi.get_pointer(),
                      1,
                      1,
                      k2d.get_p2D_global()->desc,
                      k2d.get_p2D_global()->blacs_ctxt);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        ModuleBase::timer::tick("HSolverLCAO", "collect_psi");
    }
    k2d.unset_para_env();
    ModuleBase::timer::tick("HSolverLCAO", "parakSolve");
#endif
}

template <typename T, typename Device>
void HSolverLCAO<T, Device>::parakSolve_cusolver(hamilt::Hamilt<T>* pHamilt,
                                            psi::Psi<T>& psi,
                                            elecstate::ElecState* pes)
{
    ModuleBase::timer::tick("HSolverLCAO", "parakSolve");
    const int dev_id = base_device::information::set_device_by_rank();
    int kpar = omp_get_max_threads();
    std::vector<cudaStream_t> streams(kpar);
    for(int i = 0; i < kpar; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    const int nks = psi.get_nk();  // total number of k points
    const int nbands = this->ParaV->get_nbands();
    // Set the parallel storage scheme for the matrix and psi
    Parallel_2D mat_para_global;    // store the info about how the origin matrix is distributed in parallel
    Parallel_2D mat_para_local;     // store the info about how the matrix is distributed after collected from all processes
    Parallel_2D psi_para_global;    // store the info about how the psi is distributed in parallel
    Parallel_2D psi_para_local;     // store the info about how the psi is distributed before distributing to all processes

    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, GlobalV::MY_RANK, 0, &new_comm);
    int nrow = this->ParaV->get_global_row_size(); // number of rows in the global matrix
    int ncol = nrow;
    int nb2d = this->ParaV->get_block_size();      // block size for the 2D matrix distribution
    mat_para_global.init(nrow, ncol, nb2d, MPI_COMM_WORLD);
    psi_para_global.init(nrow, nbands, nb2d, MPI_COMM_WORLD);
    mat_para_local.init(nrow, ncol, nb2d, new_comm);
    psi_para_local.init(nrow, ncol, nb2d, new_comm);
    std::vector<std::vector<T>> hk_vec;
    std::vector<std::vector<T>> sk_vec;
    for (int ik = 0; ik < nks; ik += GlobalV::NPROC * kpar)
    {
        std::vector<int> kpoints_local;   // store the k points that need to be calculated by each process in each loop

        /* store the total k points that need to be calculated in each loop
           the key is the k point index, the value is the process id that will calculate this k point */
        std::map<int, int> kpoints_global;
        for (int kpt = ik; kpt < ik + GlobalV::NPROC * kpar && kpt < nks; ++kpt)
        {
            kpoints_global[kpt] = kpt % GlobalV::NPROC;
        }

        for (int kpt = ik + GlobalV::MY_RANK; kpt < nks && kpt < ik + GlobalV::NPROC * kpar; kpt += GlobalV::NPROC)
        {
            kpoints_local.push_back(kpt);
        }

        hk_vec.resize(kpoints_local.size(), std::vector<T>(nrow * ncol, 0.0));
        sk_vec.resize(kpoints_local.size(), std::vector<T>(nrow * ncol, 0.0));
        int mat_id = 0;
        for(const auto& pair : kpoints_global )
        {
            int kpt = pair.first;
            pHamilt->updateHk(kpt);
            hamilt::MatrixBlock<T> hk_2D, sk_2D;
            pHamilt->matrix(hk_2D, sk_2D);
            int desc_tmp[9];
            T* hk_local_ptr = nullptr;
            T* sk_local_ptr = nullptr;
            std::copy(mat_para_local.desc, mat_para_local.desc + 9, desc_tmp);
            if(std::find(kpoints_local.begin(), kpoints_local.end(), kpt) == kpoints_local.end())
            {
                // if the k point is not in the local k points, set the desc[1] to -1
                // which means that the matrix will not be distributed to this process
                desc_tmp[1] = -1;
            } else
            {
                hk_local_ptr = hk_vec[mat_id].data();
                sk_local_ptr = sk_vec[mat_id].data();
                mat_id++;
            }

            Cpxgemr2d(nrow, ncol, hk_2D.p, 1, 1, mat_para_global.desc,
                      hk_local_ptr, 1, 1, desc_tmp,
                      mat_para_global.blacs_ctxt);
            Cpxgemr2d(nrow, ncol, sk_2D.p, 1, 1, mat_para_global.desc,
                      sk_local_ptr, 1, 1, desc_tmp,
                      mat_para_global.blacs_ctxt);
            
        }

        // Now we have the local hk and sk matrices, we can solve the eigenvalue problem
        std::vector<psi::Psi<T>> psi_local(kpoints_local.size(), psi::Psi<T>(1, ncol, nrow, nrow, true));
        #pragma omp parallel
        {
            cudaSetDevice(dev_id);
            #pragma omp for
            for (int i = 0; i < kpoints_local.size(); i++)
            {
                int ik = kpoints_local[i];
                psi_local[i].fix_k(0);

                hamilt::MatrixBlock<T> hk_local = hamilt::MatrixBlock<T>{
                    hk_vec[i].data(), (size_t)nrow, (size_t)ncol,
                    mat_para_local.desc};
                hamilt::MatrixBlock<T> sk_local = hamilt::MatrixBlock<T>{
                    sk_vec[i].data(), (size_t)nrow, (size_t)ncol,
                    mat_para_local.desc};
                DiagoCusolver<T> cu(nullptr, streams[i]);
                cu.diag_pool(hk_local, sk_local, psi_local[i], &(pes->ekb(ik, 0)));
            }
        }
        // Now we have the local psi matrices, we can distribute them to the global psi matrix
        for(const auto& pair: kpoints_global)
        {
            int kpt = pair.first;
            int root = pair.second;
            MPI_Bcast(&(pes->ekb(kpt, 0)), nbands, MPI_DOUBLE, root, MPI_COMM_WORLD);
            int desc_pool[9];
            std::copy(psi_para_local.desc, psi_para_local.desc + 9, desc_pool);
            auto kid = std::find(kpoints_local.begin(), kpoints_local.end(), kpt);
            T* psi_local_ptr = nullptr;
            if ( kid == kpoints_local.end())
            {
                desc_pool[1] = -1;
            }else
            {
                int psi_id = kid - kpoints_local.begin();
                psi_local_ptr = psi_local[psi_id].get_pointer();
            }
            psi.fix_k(kpt);
            Cpxgemr2d(nrow,
                      nbands,
                      psi_local_ptr,
                      1,
                      1,
                      desc_pool,
                      psi.get_pointer(),
                      1,
                      1,
                      psi_para_global.desc,
                      psi_para_global.blacs_ctxt);
        }
    }
    for(int i = 0; i < kpar; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    ModuleBase::timer::tick("HSolverLCAO", "parakSolve");
}

template class HSolverLCAO<double>;
template class HSolverLCAO<std::complex<double>>;

} // namespace hsolver