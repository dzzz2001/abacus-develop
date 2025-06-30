/*
This is the test file to run the problem defined in WeightedLowRank.h and WeightedLowRank.cpp.

---- WH
*/

#ifndef TESTFRANKETEXTUREINPAINTING_H
#define TESTFRANKETEXTUREINPAINTING_H

#include <iostream>
#include "Others/randgen.h"
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include <ctime>

#include "Problems/FRankETextureInpainting.h"
#include "Manifolds/FixedRankE.h"

#include "Solvers/RSD.h"
#include "Solvers/RNewton.h"
#include "Solvers/RCG.h"
#include "Solvers/RBroydenFamily.h"
#include "Solvers/RWRBFGS.h"
#include "Solvers/RBFGS.h"
#include "Solvers/LRBFGS.h"
#include "Solvers/IRPG.h"
#include "Solvers/IARPG.h"

#include "Solvers/RTRSD.h"
#include "Solvers/RTRNewton.h"
#include "Solvers/RTRSR1.h"
#include "Solvers/LRTRSR1.h"

#include "Others/def.h"

#include "test/DriverMexProb.h"

using namespace ROPTLIB;

void testFRankETextureInpainting(void);
Vector LADM(unsigned long *inir, unsigned long *injc, unsigned long *injcc, unsigned long innzmax, Vector inD, realdp inlambda, integer inm, integer inn, integer inr, 
        realdp mu, realdp rho, realdp eta, integer type, Vector W, integer *outIter, realdp *outComTime);

#endif
