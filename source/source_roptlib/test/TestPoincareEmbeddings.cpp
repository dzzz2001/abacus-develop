#include "test/TestPoincareEmbeddings.h"

using namespace ROPTLIB;

void testPoincareEmbeddings(void)
{
	/*Randomly generate a point on the Poincare Ball*/
	integer n = 5;
	integer numoftypes = 1, numofmani = 68; //1180 / 1109 / 68 ---
	integer sampleNum = 50;	//6540 / 2500 / 50---
	// Define the manifold
	//PoincareBall Domain(n);
	//Domain.ChooseParamsSet1();
	//Variable X = Domain.RandominManifold();
	//Domain.CheckIntrExtr(X);
	//Domain.CheckRetraction(X);
	//Domain.CheckIsometryofInvVectorTransport(X);
	//Domain.CheckIsometryofVectorTransport(X);
	//Domain.CheckVecTranComposeInverseVecTran(X);

	// Define the manifold
	PoincareBall mani(n);
	mani.ChooseParamsSet1();
	//mani.ChooseParamsSet2();
	//mani.ChooseParamsSet3();
	
	ProductManifold Domain(numoftypes, &mani, numofmani);
	Variable ProdX = Domain.RandominManifold();
	
	//Domain.CheckIntrExtr(ProdX);
	//Domain.CheckRetraction(ProdX);
	//Domain.CheckIsometryofInvVectorTransport(ProdX);
	//Domain.CheckIsometryofVectorTransport(ProdX);
	//Domain.CheckVecTranComposeInverseVecTran(ProdX);
	//start = getTickCount();
	// ------------------- read data -------------------

//    std::string fname = "data/wn_mini.csv"; //wn_mini.csv---
    std::string fname = "/Users/whuang/Documents/Syn/Codes/newROPTLIB/ROPTLIB/Matlab/ForCpp/wn_mini.csv"; //wn.csv---
	std::ifstream csv_data(fname, std::ios::in);
	std::string line;

	if (!csv_data.is_open())
	{
		std::cout << "Error: opening file fail" << std::endl;
		std::exit(1);
	}

	std::istringstream sin;
	std::string word;
	Vector words(sampleNum * 2); //6540*2 / 2500*2 / 50*2---
	realdp *wordsptr = words.ObtainWriteEntireData();
	std::getline(csv_data, line);
	integer i = 0;

	while (std::getline(csv_data, line))
	{
		sin.clear();
		sin.str(line);
		
		while (std::getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
		{
			wordsptr[i] = std::stol(word); //将每一格中的数据逐个push
			i++;
		}
	}
	csv_data.close();
	words.Reshape(2, sampleNum).Transpose(); //2,2500 / 2,50---
	

//	integer batchsize = 10; //128---
	integer NegSampleNum = 4; //10---
	realdp SampleDampening = 0.75;
	PoincareEmbeddings Prob(words, n, numofmani, NegSampleNum, SampleDampening); //---
	Prob.SetDomain(&Domain); //---
	Domain.CheckParams();
    
//    Prob.CheckGradHessian(ProdX);//--
//    return;
	
	//********************** RSGD ****************************
	RSGD *RSGDsolver = new RSGD(&Prob, &ProdX);
	RSGDsolver->Verbose = ITERRESULT;
	RSGDsolver->Max_Iteration = 200;
	RSGDsolver->Initstepsize = 1;
	RSGDsolver->Tolerance = static_cast<realdp> (1e-6);
//	RSGDsolver->Stop_Criterion = SM_FUN_PATIENCE;
//	RSGDsolver->patience = 20;
	RSGDsolver->NumFixedStep = 20;
	RSGDsolver->isFixed = true;
	RSGDsolver->burnin_multiplier = 0.01;
	RSGDsolver->StepsizeType = STO_FIXED_STEPSIZE;
	RSGDsolver->CheckParams();
	RSGDsolver->Run();

	if (RSGDsolver->Getnormgfgf0() < 1e-2)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

	delete RSGDsolver;
    
	//************************* RADAMSP ****************************
    RADAMSP *RADAMSPsolver = new RADAMSP(&Prob, &ProdX);
    RADAMSPsolver->Verbose = ITERRESULT;
    RADAMSPsolver->Max_Iteration = 200;
    RADAMSPsolver->Initstepsize = 0.1;
    RADAMSPsolver->burnin_multiplier = 3;//--0.08;
    RADAMSPsolver->NumFixedStep = 20;
    RADAMSPsolver->isFixed = false;
    RADAMSPsolver->Tolerance = static_cast<realdp> (1e-6);
//    RADAMSPsolver->Stop_Criterion = SM_FUN_PATIENCE;
//    RADAMSPsolver->patience = 20;
    RADAMSPsolver->CheckParams();
    RADAMSPsolver->Run();
	if (RADAMSPsolver->Getnormgfgf0() < 1e-2)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

	delete RADAMSPsolver;

	//************************ sparseRAMSGRAD *****************************
	RAMSGRADSP *RAMSGRADSPsolver = new RAMSGRADSP(&Prob, &ProdX);
	RAMSGRADSPsolver->Verbose = ITERRESULT;
	RAMSGRADSPsolver->Max_Iteration = 200;
	RAMSGRADSPsolver->Initstepsize = 0.1;
//	RAMSGRADSPsolver->Stop_Criterion = SM_FUN_PATIENCE;
//	RAMSGRADSPsolver->patience = 20;
	RAMSGRADSPsolver->gamma = 0.1;
	RAMSGRADSPsolver->C = 10;
	RAMSGRADSPsolver->theta = 0;
	RAMSGRADSPsolver->Tolerance = static_cast<realdp> (1e-6);
	RAMSGRADSPsolver->isFixed = true;
	RAMSGRADSPsolver->NumFixedStep = 20;
	RAMSGRADSPsolver->burnin_multiplier = 0.01;
	RAMSGRADSPsolver->CheckParams();
	RAMSGRADSPsolver->Run();

	if (RAMSGRADSPsolver->Getnormgfgf0() < 1e-2)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

	delete RAMSGRADSPsolver;


	//************************* prodManiRADAM ****************************
	/*prodManiRADAM *prodManiRADAMsolver = new prodManiRADAM(&Prob, &ProdX);
	prodManiRADAMsolver->Verbose = ITERRESULT;
	prodManiRADAMsolver->Max_Iteration = 200;
	prodManiRADAMsolver->InitStepsize = 0.08;
	prodManiRADAMsolver->burnin_multiplier = 0.1;
	prodManiRADAMsolver->FixedStep = 20;
	prodManiRADAMsolver->isFixed = false;
	prodManiRADAMsolver->Tolerance = static_cast<realdp> (1e-6);
	prodManiRADAMsolver->Stop_Criterion = SM_FUN_PATIENCE;
	prodManiRADAMsolver->patience = 20;
	prodManiRADAMsolver->CheckParams();
	prodManiRADAMsolver->Run();
	if (prodManiRADAMsolver->Getnormgfgf0() < 1e-2)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

	delete prodManiRADAMsolver;*/

	//************************* prodManiRAMSGRAD ****************************
	/*prodManiRAMSGRAD *prodManiRAMSGRADsolver = new prodManiRAMSGRAD(&Prob, &ProdX);
	prodManiRAMSGRADsolver->Verbose = ITERRESULT;
	prodManiRAMSGRADsolver->Max_Iteration = 200;
	prodManiRAMSGRADsolver->InitStepsize = 0.3;
	prodManiRAMSGRADsolver->burnin_multiplier = 0.1;
	prodManiRAMSGRADsolver->FixedStep = 20;
	prodManiRAMSGRADsolver->isFixed = false;
	prodManiRAMSGRADsolver->Tolerance = static_cast<realdp> (1e-6);
	prodManiRAMSGRADsolver->Stop_Criterion = SM_FUN_PATIENCE;
	prodManiRAMSGRADsolver->patience = 20;
	prodManiRAMSGRADsolver->CheckParams();
	prodManiRAMSGRADsolver->Run();
	if (prodManiRAMSGRADsolver->Getnormgfgf0() < 1e-2)
		printf("SUCCESS!\n");
	else
		printf("FAIL!\n");

	delete prodManiRAMSGRADsolver;*/
	
};



/*If it is compiled in Matlab, then the following "mexFunction" is used as the entrance.*/
#ifdef MATLAB_MEX_FILE

std::map<integer *, integer> *CheckMemoryDeleted;

/*This function checks the number and formats of input parameters.
nlhs: the number of output in mxArray format
plhs: the output objects in mxArray format
nrhs: the number of input in mxArray format
prhs: the input objects in mxArray format */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 7)
	{
		mexErrMsgTxt("The number of arguments should be at least seven.\n");
	}

	realdp *data, *X;
	integer n, N, NegSampleNum, ParamSet, batchsize; /* N=numofmani */
	data = mxGetPr(prhs[0]);
	integer dataM = mxGetM(prhs[0]);
	integer dataN = mxGetN(prhs[0]);
	Vector indata(dataM, dataN);
	realdp *indataptr = indata.ObtainWriteEntireData();
	for (integer i = 0; i < dataM * dataN; i++)
	{
		indataptr[i] = data[i]; 
	}

	n = static_cast<integer> (mxGetScalar(prhs[1]));
	X = mxGetPr(prhs[2]); 
    N = static_cast<integer> (mxGetScalar(prhs[3]));
    NegSampleNum = static_cast<integer> (mxGetScalar(prhs[4]));
    realdp SampleDampening = mxGetScalar(prhs[5]);

    
//	const mwSize *ptrdims = mxGetDimensions(prhs[2]);
//	if (mxGetNumberOfDimensions(prhs[2]) == 2)
//		N = 1;
//	else
//		N = ptrdims[2];
//
//	NegSampleNum = static_cast<integer> (mxGetScalar(prhs[3]));
//	ParamSet = static_cast<integer> (mxGetScalar(prhs[4]));

	genrandseed(0);
	CheckMemoryDeleted = new std::map<integer *, integer>;

	PoincareBall mani(n);
//	if (ParamSet == 1)
//		mani.ChooseParamsSet1();
//	else if (ParamSet == 2)
//		mani.ChooseParamsSet2();
//	else if(ParamSet == 3)
//		mani.ChooseParamsSet3();
//	else if(ParamSet == 4)
//		mani.ChooseParamsSet4();

	ProductManifold Domain(1, &mani, N);
//	Domain.CheckParams();

	Variable initX = Domain.RandominManifold();
	realdp *initXptr = initX.ObtainWriteEntireData();
	for (integer i = 0; i < n * N; i++)
		initXptr[i] = X[i];

//	batchsize = static_cast<integer> (mxGetScalar(prhs[6]));
//	realdp SampleDampening = mxGetScalar(prhs[7]);
	PoincareEmbeddings Prob(indata, n, N, NegSampleNum, SampleDampening);
	Prob.SetDomain(&Domain);
	ParseSolverParamsAndOptimizing(prhs[6], &Prob, &initX, plhs);

	std::map<integer *, integer>::iterator iter = CheckMemoryDeleted->begin();
	for (iter = CheckMemoryDeleted->begin(); iter != CheckMemoryDeleted->end(); iter++)
	{
		if (iter->second != 1)
			printf("Global address: %p, sharedtimes: %d\n", iter->first, iter->second);
	}
	
	delete CheckMemoryDeleted;
	return;
};

#endif
