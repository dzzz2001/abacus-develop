/*
This is the test file to run the problem defined in StieSoftICA.h and StieSoftICA.cpp.

---- WH
*/

#ifndef TESTSTIESOFTICA_H
#define TESTSTIESOFTICA_H

#include <iostream>
#include "Others/randgen.h"
#include <ctime>
#include "test/DriverMexProb.h"

#include "Problems/Problem.h"
#include "Problems/StieSoftICA.h"

#include "Manifolds/Stiefel.h"
#include "Manifolds/SphereTx.h"

#include "Solvers/RSD.h"
#include "Solvers/RNewton.h"
#include "Solvers/RCG.h"
#include "Solvers/RBroydenFamily.h"
#include "Solvers/RWRBFGS.h"
#include "Solvers/RBFGS.h"
#include "Solvers/LRBFGS.h"
#include "Solvers/RGS.h"
#include "Solvers/LRBFGSSub.h"
#include "Solvers/RBFGSSub.h"

#include "Solvers/SolversSMTR.h"
#include "Solvers/RTRSD.h"
#include "Solvers/RTRNewton.h"
#include "Solvers/RTRSR1.h"
#include "Solvers/LRTRSR1.h"

#include "Others/def.h"

using namespace ROPTLIB;
void testStieSoftICA(void);

#endif // end of TESTSTIESOFTICA_H
