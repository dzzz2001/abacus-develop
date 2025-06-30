# Makefile for ROPTLIB. Test on ubuntu 18.04 LTS

# set compiler
CC = g++

# default test problem is to check all the problems
TP?=DriverCpp

#the path of ROPTLIB
ROOTPATH:=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# set the path of Julia
JULIA_DIR:=/home/whuang/Documents/julia

# directories of ROPTLIB header files
INCDIRS = -I$(ROOTPATH)/
INCDIRS += -I$(ROOTPATH)/.vs/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/v15/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/v15/ipch/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/v15/ipch/AutoPCH/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/v15/ipch/AutoPCH/164433e706b84785/
INCDIRS += -I$(ROOTPATH)/.vs/ROPTLIB/v15/ipch/AutoPCH/4fa875e8ac0192bd/
INCDIRS += -I$(ROOTPATH)/BinaryFiles/
INCDIRS += -I$(ROOTPATH)/BinaryFiles/x64/
INCDIRS += -I$(ROOTPATH)/BinaryFiles/x64/Debug/
INCDIRS += -I$(ROOTPATH)/BinaryFiles/x64/Debug/ROPTLIB.tlog/
INCDIRS += -I$(ROOTPATH)/Manifolds/
INCDIRS += -I$(ROOTPATH)/Matlab/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/IRPG/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/IRPG/SPCA/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/IRPG/SVPCA/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/IRPG/SVPCA/PenCPG/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/IRPG/TextureInpainting/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/CFRQBlindDecon2D/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/CSFRQPhaseRetrieval/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/EucQuadratic/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/FRankE3FMatCom/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/FRankE3FMatCom/DownloadLMaFit/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/FRankE3FMatCom/DownloadLMaFit/LMaFit-adp/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/FRankE3FMatCom/DownloadLMaFit/LMaFit-adp/Utilities/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/LRTRSR1tCGvsLRTRSR1/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/LinearEigVal/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/SPDKarcherMean/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/LRTRSR1/StieSoftICA/
INCDIRS += -I$(ROOTPATH)/Matlab/ForCpp/SPCA/
INCDIRS += -I$(ROOTPATH)/Matlab/ForMatlab/
INCDIRS += -I$(ROOTPATH)/Others/
INCDIRS += -I$(ROOTPATH)/Others/SparseBLAS/
INCDIRS += -I$(ROOTPATH)/Others/fftw/
INCDIRS += -I$(ROOTPATH)/Others/wavelet/
INCDIRS += -I$(ROOTPATH)/Problems/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/xcshareddata/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/configuration/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/xcuserdata/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/project.xcworkspace/xcuserdata/whuang.xcuserdatad/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcshareddata/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcshareddata/xcschemes/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcuserdata/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcuserdata/whuang.xcuserdatad/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcuserdata/whuang.xcuserdatad/xcdebugger/
INCDIRS += -I$(ROOTPATH)/ROPTLIB_mac.xcodeproj/xcuserdata/whuang.xcuserdatad/xcschemes/
INCDIRS += -I$(ROOTPATH)/Solvers/
INCDIRS += -I$(ROOTPATH)/cwrapper/
INCDIRS += -I$(ROOTPATH)/cwrapper/blas/
INCDIRS += -I$(ROOTPATH)/cwrapper/lapack/
INCDIRS += -I$(ROOTPATH)/test/
# ROPTLIB C++ files
CPPS += $(ROOTPATH)/Manifolds/CFixedRankQ2F.cpp $(ROOTPATH)/Manifolds/CStiefel.cpp $(ROOTPATH)/Manifolds/CSymFixedRankQ.cpp $(ROOTPATH)/Manifolds/Element.cpp $(ROOTPATH)/Manifolds/Euclidean.cpp $(ROOTPATH)/Manifolds/FixedRankE.cpp $(ROOTPATH)/Manifolds/FixedRankE3F.cpp $(ROOTPATH)/Manifolds/FixedRankQ2F.cpp $(ROOTPATH)/Manifolds/Grassmann.cpp $(ROOTPATH)/Manifolds/Manifold.cpp $(ROOTPATH)/Manifolds/MultiManifolds.cpp $(ROOTPATH)/Manifolds/PoincareBall.cpp $(ROOTPATH)/Manifolds/SPDManifold.cpp $(ROOTPATH)/Manifolds/SmartSpace.cpp $(ROOTPATH)/Manifolds/Sphere.cpp $(ROOTPATH)/Manifolds/SphereTx.cpp $(ROOTPATH)/Manifolds/Stiefel.cpp $(ROOTPATH)/Manifolds/SymFixedRankQ.cpp 
CPPS += $(ROOTPATH)/Others/BlasLapackCppWrapper.cpp $(ROOTPATH)/Others/ForDebug.cpp $(ROOTPATH)/Others/MinPNormConHull.cpp $(ROOTPATH)/Others/SparseMatrix.cpp $(ROOTPATH)/Others/Spline.cpp $(ROOTPATH)/Others/Timer.cpp $(ROOTPATH)/Others/randgen.cpp 
CPPS += $(ROOTPATH)/Others/SparseBLAS/nist_spblas.cpp 
CPPS += $(ROOTPATH)/Others/wavelet/wavelet.cpp 
CPPS += $(ROOTPATH)/Problems/CFRankQ2FBlindDecon2D.cpp $(ROOTPATH)/Problems/CSFRQPhaseRetrieval.cpp $(ROOTPATH)/Problems/CStieBrockett.cpp $(ROOTPATH)/Problems/EucQuadratic.cpp $(ROOTPATH)/Problems/FRankE3FMatCompletion.cpp $(ROOTPATH)/Problems/FRankESparseApprox.cpp $(ROOTPATH)/Problems/FRankETextureInpainting.cpp $(ROOTPATH)/Problems/FRankEWeightApprox.cpp $(ROOTPATH)/Problems/FRankQ2FMatCompletion.cpp $(ROOTPATH)/Problems/GrassMatCompletion.cpp $(ROOTPATH)/Problems/GrassPCA.cpp $(ROOTPATH)/Problems/GrassRQ.cpp $(ROOTPATH)/Problems/GrassSVPCA.cpp $(ROOTPATH)/Problems/ObliqueSPCA.cpp $(ROOTPATH)/Problems/PoincareEmbeddings.cpp $(ROOTPATH)/Problems/Problem.cpp $(ROOTPATH)/Problems/ProdStieSumBrockett.cpp $(ROOTPATH)/Problems/SFRQLyapunov.cpp $(ROOTPATH)/Problems/SPDKarcherMean.cpp $(ROOTPATH)/Problems/SphereConvexHull.cpp $(ROOTPATH)/Problems/SphereSparsestVector.cpp $(ROOTPATH)/Problems/SphereTxRQ.cpp $(ROOTPATH)/Problems/StieBrockett.cpp $(ROOTPATH)/Problems/StieSPCA.cpp $(ROOTPATH)/Problems/StieSoftICA.cpp $(ROOTPATH)/Problems/juliaProblem.cpp $(ROOTPATH)/Problems/mexProblem.cpp 
CPPS += $(ROOTPATH)/Solvers/IARPG.cpp $(ROOTPATH)/Solvers/IRPG.cpp $(ROOTPATH)/Solvers/LRBFGS.cpp $(ROOTPATH)/Solvers/LRBFGSSub.cpp $(ROOTPATH)/Solvers/LRBroydenFamily.cpp $(ROOTPATH)/Solvers/LRTRSR1.cpp $(ROOTPATH)/Solvers/LRTRSR1woR.cpp $(ROOTPATH)/Solvers/RADAM.cpp $(ROOTPATH)/Solvers/RADAMSP.cpp $(ROOTPATH)/Solvers/RAMSGRAD.cpp $(ROOTPATH)/Solvers/RAMSGRADSP.cpp $(ROOTPATH)/Solvers/RBFGS.cpp $(ROOTPATH)/Solvers/RBFGSSub.cpp $(ROOTPATH)/Solvers/RBroydenFamily.cpp $(ROOTPATH)/Solvers/RCG.cpp $(ROOTPATH)/Solvers/RGS.cpp $(ROOTPATH)/Solvers/RNewton.cpp $(ROOTPATH)/Solvers/RSD.cpp $(ROOTPATH)/Solvers/RSGD.cpp $(ROOTPATH)/Solvers/RSVRG.cpp $(ROOTPATH)/Solvers/RTRNewton.cpp $(ROOTPATH)/Solvers/RTRSD.cpp $(ROOTPATH)/Solvers/RTRSR1.cpp $(ROOTPATH)/Solvers/RWRBFGS.cpp $(ROOTPATH)/Solvers/SVRLRBFGS.cpp $(ROOTPATH)/Solvers/SVRLRBroydenFamily.cpp $(ROOTPATH)/Solvers/Solvers.cpp $(ROOTPATH)/Solvers/SolversNSM.cpp $(ROOTPATH)/Solvers/SolversNSMPGLS.cpp $(ROOTPATH)/Solvers/SolversNSMSub.cpp $(ROOTPATH)/Solvers/SolversNSMSubLS.cpp $(ROOTPATH)/Solvers/SolversSM.cpp $(ROOTPATH)/Solvers/SolversSMLS.cpp $(ROOTPATH)/Solvers/SolversSMSVRG.cpp $(ROOTPATH)/Solvers/SolversSMSto.cpp $(ROOTPATH)/Solvers/SolversSMTR.cpp 
CPPS += $(ROOTPATH)/test/TestCFRankQ2FBlindDecon2D.cpp $(ROOTPATH)/test/TestCSFRQPhaseRetrieval.cpp $(ROOTPATH)/test/TestCStieBrockett.cpp $(ROOTPATH)/test/TestElement.cpp $(ROOTPATH)/test/TestEucQuadratic.cpp $(ROOTPATH)/test/TestFRankE3FMatCompletion.cpp $(ROOTPATH)/test/TestFRankESparseApprox.cpp $(ROOTPATH)/test/TestFRankETextureInpainting.cpp $(ROOTPATH)/test/TestFRankEWeightApprox.cpp $(ROOTPATH)/test/TestFRankQ2FMatCompletion.cpp $(ROOTPATH)/test/TestGrassMatCompletion.cpp $(ROOTPATH)/test/TestGrassPCA.cpp $(ROOTPATH)/test/TestGrassRQ.cpp $(ROOTPATH)/test/TestGrassSVPCA.cpp $(ROOTPATH)/test/TestObliqueSPCA.cpp $(ROOTPATH)/test/TestPoincareEmbeddings.cpp $(ROOTPATH)/test/TestProdStieSumBrockett.cpp $(ROOTPATH)/test/TestSFRQLyapunov.cpp $(ROOTPATH)/test/TestSPDKarcherMean.cpp $(ROOTPATH)/test/TestSphereSparsestVector.cpp $(ROOTPATH)/test/TestStieBrockett.cpp $(ROOTPATH)/test/TestStieSPCA.cpp $(ROOTPATH)/test/TestStieSoftICA.cpp 
# convert a string to upper case.
UPPER_TP  = $(shell echo $(TP) | tr a-z A-Z)

# make a binary file, which is called in command line
ROPTLIB:
	$(CC) -O3 -w -std=c++0x $(ROOTPATH)/test/$(TP).cpp $(CPPS) $(INCDIRS) -D$(UPPER_TP) -DROPTLIB_WITH_FFTW -llapack -lblas -lfftw3 -lm -o $(TP)

#make a library:
libropt.so:
	$(CC) -w -std=c++0x -shared -fPIC -O3 $(CPPS) $(INCDIRS) -DROPTLIB_WITH_FFTW -llapack -lblas -lfftw3 -lm -o $@

#make a julia library:
JULIA_LIB:=$(JULIA_DIR)/lib
JULIA_SRC:=$(JULIA_DIR)/src
JULIA_INC:=$(JULIA_DIR)/include/julia
CPPFLAGS:=-I$(JULIA_INC) -I$(JULIA_SRC) -I$(JULIA_SRC)/support
LDFLAGS:=-L$(JULIA_LIB)
LDLIBS=-ljulia
export LD_LIBRARY_PATH:=$(JULIA_LIB):$(JULIA_LIB)/julia

JuliaROPTLIB:
	$(CC) -O3 -shared -fPIC -std=c++0x $(ROOTPATH)/test/$(TP).cpp $(CPPS) $(INCDIRS) -D$(UPPER_TP) $(CPPFLAGS) $(LDFLAGS) -Wl,-rpath,$(JULIA_LIB) -lm $(LDLIBS) -DJULIA_LIB_DIR="$(JULIA_DIR)/lib/julia" -DROPTLIB_WITH_FFTW -llapack -lblas -lfftw3 -o $(TP).so
