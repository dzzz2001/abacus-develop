
    seed = floor(rand() * 100000);
    seed = 1;
    fprintf('CTestPoincareEmbeddings seed:%d\n', seed);
    rng(seed);

    n = 5;
    numofmani = 68;
    NegSampleNum = 4;
    SampleDampening = 0.75;

    % wn_mini.csv
    words = [63,44,13,4,60,8,36,52,43,54,30,40,66,47,56,35,38,0,29,17,57,65,46,41,24,55,23,42,14,39,6,51,49,28,16,34,61,53,25,18,21,32,59,20,67,64,33,19,1,5;
             9,37,15,37,27,10,45,22,10,45,37,26,50,58,62,11,45,10,62,37,45,9,37,12,37,45,37,45,62,10,45,37,48,58,37,45,37,3,48,45,10,7,45,27,15,45,31,2,48,6];
    words = words';

    SolverParams.method = 'RSGD';
    SolverParams.Initstepsize = 3;

    % SolverParams.method = 'RSVRG';
    % SolverParams.Initstepsize = 1;
    % 
    % SolverParams.method = 'SVRLRBFGS';
    % SolverParams.Initstepsize = 0.005;
    % 
    % SolverParams.method = 'RADAM';
    % SolverParams.Initstepsize = 0.08;

    SolverParams.IsCheckParams = 1;
    SolverParams.Max_Iteration = 5000;
    X = (rand(n, numofmani) - 0.5) * 0.002;

    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = TestPoincareEmbeddings(words, n, X, numofmani, NegSampleNum, SampleDampening, SolverParams);
