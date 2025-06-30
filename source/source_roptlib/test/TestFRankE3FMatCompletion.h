/*
This is the test file to run the problem defined in LRMatrixCompletion.h and LRMatrixCompletion.cpp.

---- WH
*/

#ifndef TESTFRANKE3FMATCOMPLETION_H
#define TESTFRANKE3FMATCOMPLETION_H


#include <iostream>
#include "Others/randgen.h"
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include <ctime>

#include "test/DriverMexProb.h"

#include "Problems/FRankE3FMatCompletion.h"
#include "Manifolds/FixedRankE3F.h"

#include "Solvers/RSD.h"
#include "Solvers/RNewton.h"
#include "Solvers/RCG.h"
#include "Solvers/RBroydenFamily.h"
#include "Solvers/RWRBFGS.h"
#include "Solvers/RBFGS.h"
#include "Solvers/LRBFGS.h"

#include "Solvers/RTRSD.h"
#include "Solvers/RTRNewton.h"
#include "Solvers/RTRSR1.h"
#include "Solvers/LRTRSR1.h"

#include "Others/def.h"

using namespace ROPTLIB;

void testFRankE3FMatCompletion(void);
realdp LRMatrixCompletionLinesearchInput(integer iter, const Variable &x1, const Vector &eta1, realdp initialstepsize, realdp initialslope, const Problem *prob, const Solvers *solver);

#endif
