
    seed = floor(rand() * 100000);
    seed = 2;
    fprintf('CTestGrassPCA seed:%d\n', seed);
    rng(seed);
    n = 3;
    p = 2;
    N = 50;
    A = randn(n, N);
    Xinitial = orth(randn(n, p));
    SolverParams.method = 'LRBFGS';
%     SolverParams.method = 'RTRSR1';
%     SolverParams.method = 'RTRNewton';

    SolverParams.method = 'RSGD';
    SolverParams.Initstepsize = 0.002;

    SolverParams.method = 'RSVRG';
    SolverParams.Initstepsize = 0.002;

    SolverParams.method = 'SVRLRBFGS';
    SolverParams.Initstepsize = 0.002;

    SolverParams.method = 'RADAM';
    SolverParams.Initstepsize = 0.008;

    SolverParams.IsCheckParams = 1;
    SolverParams.Max_Iteration = 3000;
    % SolverParams.LengthSY = 4;
    % SolverParams.Verbose = 1;
    % SolverParams.Accuracy = 1e-6;
    SolverParams.Tolerance = 1e-2;
%     SolverParams.Finalstepsize = 1;
%     SolverParams.InitSteptype = 3;
%     SolverParams.IsCheckGradHess = 1;
    HasHHR = 1;
    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime] = TestGrassPCA(A, Xinitial, HasHHR, SolverParams);

