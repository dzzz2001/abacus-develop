
    seed = floor(rand() * 100000);
    seed = 20;
    fprintf('CTestObliqueSPCA seed:%d\n', seed);
    rng(seed);
    
    n = 2000;
    m = 50;
    p = 5;
    A = randn(m, n);
    A = A - repmat(mean(A, 1), m, 1);
    A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
    
    [U, S, V] = svd(A, 'econ');
    PCAV = V(:, 1:p);
    Xinitial = PCAV; % n * p
    tmp = A * PCAV; 
    maxvar = sum(tmp(:) .* tmp(:));
    
    lambda = 1;
    
    SolverParams.method = 'IRPG';
    SolverParams.IsCheckParams = 1;
    SolverParams.RPGVariant = 1; %0: IRPG without adaptive stepsize, 1: IRPG with adaptive stepsize, 3: IRPG with BB stepsize
    SolverParams.OutputGap = 20;
    SolverParams.Max_Iteration = 100;
    SolverParams.Verbose = 2;
    [xopt, fManPG, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = ...
        TestObliqueSPCA(A, lambda, Xinitial, 0, SolverParams);
    
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * p);
    xopt = reshape(xopt.main, n, p);
    % adjusted variance
    [Q, R] = qr(A * xopt, 0);
    avar = trace(R * R);
    relavar = avar / maxvar;
    fprintf('sparsity:%e, relavar:%e\n', sparsity, relavar);
    
    