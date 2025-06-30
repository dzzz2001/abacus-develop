% function MTestLRMatrixCompletion()
    seed = floor(rand() * 100000);
%     seed = 26052;
%     seed = 1;
    fprintf('CTestFRankE3FMatCompletion seed:%d\n', seed);
    rng(seed);
    m = 200;
    n = 200;
    r = 4;
    
    G = randn(m, r);
    H = randn(n, r);
%     B = G * H';
    OS = 2.5;
%     [(m + n - r) * r * OS, m * n]
    nz = min((m + n - r) * r * OS, m * n);
    
    vidx = randperm(m * n, nz);
    [ir, jc] = ind2sub([m, n], vidx);
    
    Bv = zeros(1, length(ir));
    for i = 1 : length(Bv)
        Bv(i) = G(ir(i), :) * H(jc(i), :)';
    end
%     norm(Bv - B(vidx))
    
    A = sparse(ir, jc, Bv);

%     tic
%     [U, D, V] = svds(@(x, tflag)Afun(x, tflag, A), [m, n], r);
%     toc
    U = randn(m, r);
    D = randn(r, r);
    V = randn(n, r);
    fprintf('Get leading singular vectors of A\n');
    Xinitial = [U(:); D(:); V(:)];
    SolverParams.method = 'LRBFGS';
%     SolverParams.method = 'RTRSR1';
%     SolverParams.method = 'RTRNewton';
    SolverParams.IsCheckParams = 1;
    SolverParams.Max_Iteration = 500;
    SolverParams.OutputGap = 10;
    SolverParams.Verbose = 2;
%     SolverParams.InitSteptype = 0;
    HasHHR = 0;
    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, Heigs] = TestFRankE3FMatCompletion(A, Xinitial, r, HasHHR, SolverParams);
    Heigs
% end

function output = Afun(x, tflag, A)
    if strcmp(tflag,'notransp')
        output = A * x;
    else
        output = A' * x;
    end
end
