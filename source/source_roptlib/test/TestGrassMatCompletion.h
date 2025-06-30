/*
 This is the test file to run the problem defined in LRMatrixCompletion.h and LRMatrixCompletion.cpp.
 
 ---- WH
 */

#ifndef TESTGRASSMATCOMPLETION_H
#define TESTGRASSMATCOMPLETION_H


#include <iostream>
#include "Others/randgen.h"
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include "Solvers/SolversSMSVRG.h"
#include <ctime>

/*If this test file is called from Matlab, then functions in DriverMexProb.h are used.*/
#include "test/DriverMexProb.h"

//#include "Manifolds/Euclidean/EucVariable.h"
#include "Problems/GrassMatCompletion.h"
#include "Manifolds/Grassmann.h"
//#include "Problems/StieBrockett/StieBrockett.h"
//#include "Manifolds/Stiefel/StieVector.h"
//#include "Manifolds/Stiefel/StieVariable.h"
//#include "Manifolds/Stiefel/Stiefel.h"

#include "Solvers/RADAM.h"
#include "Solvers/RADAMSP.h"
#include "Solvers/RAMSGRAD.h"
#include "Solvers/RAMSGRADSP.h"
#include "Solvers/RSD.h"
#include "Solvers/RSGD.h"
#include "Solvers/RNewton.h"
#include "Solvers/RCG.h"
#include "Solvers/RBroydenFamily.h"
#include "Solvers/RWRBFGS.h"
#include "Solvers/RBFGS.h"
#include "Solvers/LRBFGS.h"
#include "Solvers/LRBroydenFamily.h"
//#include "Solvers/LRconstant.h"
#include "Solvers/RSVRG.h"
#include "Solvers/SVRLRBFGS.h"
//#include "Solvers/SVRLRBFGS2.h"
#include "Solvers/SVRLRBroydenFamily.h"
//#include "Solvers/SVRLRBroyden2.h"

#include "Solvers/SolversSMTR.h"
#include "Solvers/RTRSD.h"
#include "Solvers/RTRNewton.h"
#include "Solvers/RTRSR1.h"
#include "Solvers/LRTRSR1.h"

#include "Others/def.h"


using namespace ROPTLIB;

void testGrassMatCompletion(void);

#endif
