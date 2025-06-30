/*
This is the test file to check all problems.

---- WH
*/

#ifndef DRIVERCPP_H
#define DRIVERCPP_H

#include "Others/def.h"

//#include "test/TestCFR2BlindDecon2D.h"
//#include "test/TestCFR2BlindDeconvolution.h"
//#include "test/TestCSO.h"
//#include "test/TestCSOPhaseRetrieval.h"
//#include "test/TestElasticCurvesRO.h"
//#include "test/TestEucBlindDeconvolution.h"
//#include "test/TestEucFrechetMean.h"
//#include "test/TestEucPosSpCd.h"
#include "test/TestCFRankQ2FBlindDecon2D.h"
#include "test/TestCSFRQPhaseRetrieval.h"
#include "test/TestCStieBrockett.h"
#include "test/TestElement.h"
#include "test/TestEucQuadratic.h"
#include "test/TestFRankE3FMatCompletion.h" /*example of sparse matrix operations*/
#include "test/TestFRankESparseApprox.h" /* example of proximal gradient method on the fixed-rank matrix manifold */
#include "test/TestFRankETextureInpainting.h" /* example of proximal gradient method on the fixed-rank matrix manifold */
#include "test/TestFRankEWeightApprox.h"
#include "test/TestFRankQ2FMatCompletion.h" /*example of sparse matrix operations*/
#include "test/TestGrassMatCompletion.h"
#include "test/TestGrassPCA.h"
#include "test/TestGrassRQ.h"
#include "test/TestGrassSVPCA.h"
//#include "test/TestKarcherMean.h"
//#include "test/TestLRBlindDeconvolution.h"
//#include "test/TestLRMatrixCompletion.h"
//#include "test/TestLRSparseMatrixApprox.h"
//#include "test/TestMNCRModel.h"
//#include "test/TestMyMatrix.h"
//#include "test/TestNSOLyapunov.h"
//#include "test/TestObliqueQCor.h"
#include "test/TestObliqueSPCA.h"
//#include "test/TestOrthBoundingBox.h"
//#include "test/TestPreShapePathStraighten.h"
#include "test/TestPoincareEmbeddings.h"
//#include "test/TestProduct.h"
#include "test/TestProdStieSumBrockett.h"
#include "test/TestSFRQLyapunov.h"
//#include "test/TestShapePathStraighten.h"
//#include "test/TestSparsePCA.h"
#include "test/TestSPDKarcherMean.h"
//#include "test/TestSPDMeanL1.h"
//#include "test/TestSPDMeanLinfty.h"
//#include "test/TestSPDMeanLDOneParam.h"
//#include "test/TestSPDMeanLDOneParamL1.h"
//#include "test/TestSPDMeanLDOneParamLinfty.h"
//#include "test/TestSPDMeanSymmLDOneParam.h"
//#include "test/TestSPDMeanSymmLDOneParamL1.h"
//#include "test/TestSPDMeanSymmLDOneParamLinfty.h"
//#include "test/TestSPDMeanMajorization.h"
//#include "test/TestSPDTensorDL.h"
//#include "test/TestSphereRayQuo.h"
#include "test/TestSphereSparsestVector.h" /* example of subgradient-based methods for nonsmooth optimization*/
#include "test/TestStieBrockett.h"
#include "test/TestStieSoftICA.h"
//#include "test/TestStieSparseBrockett.h"
#include "test/TestStieSPCA.h" /* example of proximal gradient method on the Stiefel manifold */
//#include "test/TestTestSparsePCA.h"
//#include "test/TestWeightedLowRank.h"

//#include "Problems/CSFRQPhaseRetrieval.h"

using namespace ROPTLIB;

void testall(void);
/*numGradHess: 0, test both given gradient/Hessian and numerical gradient/Hessian
               1, test only given gradient/Hessian
               2, test only numerical gradient/Hessian */
void testSmoothProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, integer numGradHess = 0);
void testStoSmoothProblem(Problem *Prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, realdp initstepsizes[], integer numGradHess = 0);
void testProxGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames);
void testSubGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames);
bool stringinclude(std::vector<std::string> names, std::string name);
int main(void);



#endif
