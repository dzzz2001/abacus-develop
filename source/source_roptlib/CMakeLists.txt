

add_library(
   roptlib
    OBJECT
Manifolds/CFixedRankQ2F.cpp
Manifolds/CStiefel.cpp
Manifolds/CSymFixedRankQ.cpp
Manifolds/Element.cpp
Manifolds/Euclidean.cpp
Manifolds/FixedRankE.cpp
Manifolds/FixedRankE3F.cpp
Manifolds/FixedRankQ2F.cpp
Manifolds/Grassmann.cpp
Manifolds/Manifold.cpp
Manifolds/MultiManifolds.cpp
Manifolds/PoincareBall.cpp
Manifolds/SPDManifold.cpp
Manifolds/SmartSpace.cpp
Manifolds/Sphere.cpp
Manifolds/SphereTx.cpp
Manifolds/Stiefel.cpp
Manifolds/SymFixedRankQ.cpp

Others/BlasLapackCppWrapper.cpp
Others/ForDebug.cpp
Others/MinPNormConHull.cpp
Others/SparseMatrix.cpp
Others/Spline.cpp
Others/Timer.cpp
Others/randgen.cpp

Others/SparseBLAS/nist_spblas.cpp

Others/wavelet/wavelet.cpp

Problems/CFRankQ2FBlindDecon2D.cpp
Problems/CSFRQPhaseRetrieval.cpp
Problems/CStieBrockett.cpp
Problems/EucQuadratic.cpp
Problems/FRankE3FMatCompletion.cpp
Problems/FRankESparseApprox.cpp
Problems/FRankETextureInpainting.cpp
Problems/FRankEWeightApprox.cpp
Problems/FRankQ2FMatCompletion.cpp
Problems/GrassMatCompletion.cpp
Problems/GrassPCA.cpp
Problems/GrassRQ.cpp
Problems/GrassSVPCA.cpp
Problems/ObliqueSPCA.cpp
Problems/PoincareEmbeddings.cpp
Problems/Problem.cpp
Problems/ProdStieSumBrockett.cpp
Problems/SFRQLyapunov.cpp
Problems/SPDKarcherMean.cpp
Problems/SphereConvexHull.cpp
Problems/SphereSparsestVector.cpp
Problems/SphereTxRQ.cpp
Problems/StieBrockett.cpp
Problems/StieSPCA.cpp
Problems/StieSoftICA.cpp
Problems/juliaProblem.cpp
Problems/mexProblem.cpp

Solvers/IARPG.cpp
Solvers/IRPG.cpp
Solvers/LRBFGS.cpp
Solvers/LRBFGSSub.cpp
Solvers/LRBroydenFamily.cpp
Solvers/LRTRSR1.cpp
Solvers/LRTRSR1woR.cpp
Solvers/RADAM.cpp
Solvers/RADAMSP.cpp
Solvers/RAMSGRAD.cpp
Solvers/RAMSGRADSP.cpp
Solvers/RBFGS.cpp
Solvers/RBFGSSub.cpp
Solvers/RBroydenFamily.cpp
Solvers/RCG.cpp
Solvers/RGS.cpp
Solvers/RNewton.cpp
Solvers/RSD.cpp
Solvers/RSGD.cpp
Solvers/RSVRG.cpp
Solvers/RTRNewton.cpp
Solvers/RTRSD.cpp
Solvers/RTRSR1.cpp
Solvers/RWRBFGS.cpp
Solvers/SVRLRBFGS.cpp
Solvers/SVRLRBroydenFamily.cpp
Solvers/Solvers.cpp
Solvers/SolversNSM.cpp
Solvers/SolversNSMPGLS.cpp
Solvers/SolversNSMSub.cpp
Solvers/SolversNSMSubLS.cpp
Solvers/SolversSM.cpp
Solvers/SolversSMLS.cpp
Solvers/SolversSMSVRG.cpp
Solvers/SolversSMSto.cpp
Solvers/SolversSMTR.cpp

# test/TestCFRankQ2FBlindDecon2D.cpp
# test/TestCSFRQPhaseRetrieval.cpp
# test/TestCStieBrockett.cpp
# test/TestElement.cpp
# test/TestEucQuadratic.cpp
# test/TestFRankE3FMatCompletion.cpp
# test/TestFRankESparseApprox.cpp
# test/TestFRankETextureInpainting.cpp
# test/TestFRankEWeightApprox.cpp
# test/TestFRankQ2FMatCompletion.cpp
# test/TestGrassMatCompletion.cpp
# test/TestGrassPCA.cpp
# test/TestGrassRQ.cpp
# test/TestGrassSVPCA.cpp
# test/TestObliqueSPCA.cpp
# test/TestPoincareEmbeddings.cpp
# test/TestProdStieSumBrockett.cpp
# test/TestSFRQLyapunov.cpp
# test/TestSPDKarcherMean.cpp
# test/TestSphereSparsestVector.cpp
# test/TestStieBrockett.cpp
# test/TestStieSPCA.cpp
# test/TestStieSoftICA.cpp

)

target_include_directories(roptlib
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/Manifolds
      ${CMAKE_CURRENT_SOURCE_DIR}/Others
      ${CMAKE_CURRENT_SOURCE_DIR}/Others/SparseBLAS
      ${CMAKE_CURRENT_SOURCE_DIR}/Others/fftw
      ${CMAKE_CURRENT_SOURCE_DIR}/Others/wavelet
      ${CMAKE_CURRENT_SOURCE_DIR}/Problems
      ${CMAKE_CURRENT_SOURCE_DIR}/Solvers
      ${CMAKE_CURRENT_SOURCE_DIR}/cwrapper
      ${CMAKE_CURRENT_SOURCE_DIR}/cwrapper/blas
      ${CMAKE_CURRENT_SOURCE_DIR}/cwrapper/lapack
      # ${CMAKE_CURRENT_SOURCE_DIR}/test
)



# if(ENABLE_COVERAGE)
#   add_coverage(surchem)
# endif()

# IF (BUILD_TESTING)
#   if(ENABLE_MPI)
#     add_subdirectory(test)
#   endif()
# endif()
