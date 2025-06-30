/*
This is the test file for the Poincare Embeddings defined in PoincareEmbeddings.h and PoincareEmbeddings.cpp.

---- YHH
*/

#ifndef TESTPOINCAREEMBEDDINGS_H
#define TESTPOINCAREEMBEDDINGS_H

/*Output to console*/
#include <iostream>

/*Generate random number*/
#include "Others/randgen.h"

/*Computational time*/
#include <ctime>

/*If this test file is called from Matlab, then functions in DriverMexProb.h are used.*/
#include "test/DriverMexProb.h"

/*Problem related classes*/
#include "Problems/Problem.h"
#include "Problems/PoincareEmbeddings.h"

/*Manifold related classes*/
#include "Manifolds/Manifold.h"
#include "Manifolds/PoincareBall.h"

/*Solvers*/
#include "Solvers/RSGD.h"
#include "Solvers/RADAM.h"
#include "Solvers/RADAMSP.h"
#include "Solvers/RAMSGRAD.h"
#include "Solvers/RAMSGRADSP.h"
//#include "Solvers/sparseRADAM.h"
//#include "Solvers/sparseRAMSGRAD.h"
//#include "Solvers/prodManiRADAM.h"

/*The global head file*/
#include "Others/def.h"
#include "Others/Timer.h"

#include<random>
#include <string>
#include <fstream>

using namespace ROPTLIB;

/*The main test function*/
void testPoincareEmbeddings(void);


#endif // end of TESTPOINCAREEMBEDDINGS_H
