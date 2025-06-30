    seed = floor(rand() * 100000);
    fprintf('CTestGrassMatCompletion seed:%d\n', seed);
    rng(seed);
    d = 10;
    n = 50;
    r = 2;
    G = randn(d, r);
    H = randn(n, r);
    OS = 3;
    nz = min((d + n - r) * r * OS, d * n);
    
    vidx = randperm(d * n, nz);
    [ir, jc] = ind2sub([d, n], vidx);
    
    Bv = zeros(1, length(ir));
    for i = 1 : length(Bv)
        Bv(i) = G(ir(i), :) * H(jc(i), :)';
    end
    A = sparse(ir, jc, Bv);

    Xinitial = orth(randn(d, r));
    % SolverParams.method = 'LRBFGS';
    % SolverParams.method = 'RTRSR1';
    % SolverParams.method = 'RTRNewton';
    SolverParams.method = 'RSGD';
    SolverParams.Initstepsize = 0.01;

    SolverParams.method = 'RSVRG';
    SolverParams.Initstepsize = 0.01;

    SolverParams.method = 'SVRLRBFGS';
    SolverParams.Initstepsize = 0.01;

    SolverParams.method = 'RADAM';
    SolverParams.Initstepsize = 0.1;
    
    SolverParams.IsCheckParams = 1;
    SolverParams.Max_Iteration = 3000;
    SolverParams.OutputGap = 50;
    SolverParams.Verbose = 2;
    HasHHR = 0;
    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, Heigs] = TestGrassMatCompletion(A, Xinitial, HasHHR, SolverParams);
