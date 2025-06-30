
    seed = floor(rand() * 100000);
    seed = 20;
    fprintf('CTestGrassSVPCA seed:%d\n', seed);
    rng(seed);
    
    n = 2000;
    m = 100;
    p = 4;
    A = randn(m, n);
    A = A - repmat(mean(A, 1), m, 1);
    A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
    
    [U, S, V] = svd(A, 'econ');
    PCAV = V(:, 1:p);
    Xinitial = PCAV; % n * p
    tmp = A * PCAV; 
    maxvar = sum(tmp(:) .* tmp(:));
    
    lambda = 0.1 * sqrt(p + log(n));
    
    SolverParams.method = 'IRPG';
    SolverParams.IsCheckParams = 1;
    SolverParams.RPGVariant = 0; %0: RPG without adaptive stepsize, 1: RPG with adaptive stepsize
    SolverParams.OutputGap = 20;
    SolverParams.Max_Iteration = 1000;
    SolverParams.Verbose = 1;
    SolverParams.Tolerance = 1e-4;
    [xopt, fManPG, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = ...
        TestGrassSVPCA(A, lambda, Xinitial, SolverParams);
    
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * p);
    xopt = reshape(xopt.main, n, p);
    % adjusted variance
    [Q, R] = qr(A * xopt, 0);
    avar = trace(R * R);
    relavar = avar / maxvar;
    fprintf('sparsity:%e, relavar:%e\n', sparsity, relavar);
    