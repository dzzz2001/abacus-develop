#add_subdirectory(kernels/cuda)
if(ENABLE_LCAO)

list(APPEND objects
    biggrid_info.cpp
    big_grid.cpp
    divide_info.cpp
    gint_atom.cpp
    gint_info.cpp
    gint.cpp
    gint_vl.cpp
    gint_vl_metagga.cpp
    gint_vl_nspin4.cpp
    gint_vl_metagga_nspin4.cpp
    gint_rho.cpp
    gint_tau.cpp
    gint_fvl.cpp
    gint_fvl_meta.cpp
    gint_env_gamma.cpp
    gint_env_k.cpp
    gint_dvlocal.cpp
    localcell_info.cpp
    phi_operator.cpp
    set_ddphi.cpp
    unitcell_info.cpp
    gint_common.cpp
    gint_interface.cpp
    )
if(USE_CUDA)
  list(APPEND objects
      kernel/gint_gpu_vars.cpp
      kernel/phi_operator_gpu.cu
      kernel/phi_operator_kernel.cu
      kernel/set_const_mem.cu
      batch_biggrid.cpp
      gint_vl_gpu.cpp
      gint_rho_gpu.cpp
      gint_fvl_gpu.cpp
      gint_vl_metagga_gpu.cpp
      gint_vl_nspin4_gpu.cpp
      gint_vl_metagga_nspin4_gpu.cpp
      gint_tau_gpu.cpp
      gint_fvl_meta_gpu.cpp
      kernel/dgemm_vbatch.cu
  )
endif()

add_library(
    gint
    OBJECT
    ${objects}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# Standalone CUDA microbench for the batched-GEMM dispatchers
# (gemm_nn_vbatch / gemm_tn_vbatch). Built whenever USE_CUDA is on, so it's
# available without BUILD_TESTING. Compiles dgemm_vbatch.cu into its own
# translation unit so it doesn't link against the full gint library — keeps
# iteration cheap and isolates the kernel under test.
if(USE_CUDA)
  add_executable(gint_vbatch_microbench
      test/test_gemm_vbatch.cu
      kernel/dgemm_vbatch.cu)
  target_include_directories(gint_vbatch_microbench
      PRIVATE ${CMAKE_SOURCE_DIR}/source)
  target_link_libraries(gint_vbatch_microbench PRIVATE CUDA::cudart)
  set_target_properties(gint_vbatch_microbench PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(ENABLE_COVERAGE)
  add_coverage(gint)
endif()

endif()
