% function MTestStieSPCA()
% load('testimage.mat');
    clear;
    seed = floor(rand() * 100000);
    seed = 20;
    fprintf('CTestFRankESparseApprox seed:%d\n', seed);
    rng(seed);
    
    A = double(imread('cameraman.tif'));
    A = haarFWT_2d(A);
%     
%     [U, S, V] = svd(A);
%     lowrankim = U(:, 1:50) * S(1:50, 1:50) * V(:, 1:50)';
%     lowrankim(find(abs(lowrankim) < 10)) = 0;
%     sum(sum(lowrankim == 0)) / prod(size(lowrankim))
%     imagesc(haarFWT_2d_inverse(lowrankim));
%     
%     return
%     A = A / norm(A);
    [m, n] = size(A);
    r = 10;
    
%     n = 10;
%     m = 10;
%     r = 5;
%     A = randn(m, n);
    
    
    U = orth(randn(m, r));
    D = randn(r, r);
    V = orth(randn(n, r));
    Xinitial.main = U * D * V';
    Xinitial.U = U;
    Xinitial.D = D;
    Xinitial.V = V;
    lambda = 1;
    
    SolverParams.method = 'IRPG';
    SolverParams.IsCheckParams = 1;
%     SolverParams.RPGVariant = 0; %0: RPG without adaptive stepsize, 1: RPG with adaptive stepsize
    SolverParams.LengthW = 1;
    SolverParams.OutputGap = 10;
    SolverParams.Max_Iteration = 1000;
    SolverParams.Min_Iteration = 10;
    SolverParams.DEBUG = 1;
    [xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = ...
        TestFRankESparseApprox(A, lambda, Xinitial, SolverParams);
    
    
% end

function M = haarFWT_2d(M)
    [n1, n2] = size(M);
    tmp = M;
    r2 = sqrt(2);
    k = 1;
    while(2 * k <= n1)
        k = k * 2;
    end
    while(1 < k)
        k = k / 2;
        for j = 1 : n2
            for i = 1 : k
                tmp(i, j) = (M(2 * i - 1, j) + M(2 * i, j)) / r2;
                tmp(k + i, j) = (M(2 * i - 1, j) - M(2 * i, j)) / r2;
            end
        end
        for j = 1 : n2
            for i = 1 : 2 * k
                M(i, j) = tmp(i, j);
            end
        end
    end
    k = 1;
    while(2 * k <= n2)
         k = k * 2;
    end
    while(1 < k)
        k = k / 2;
        for j = 1 : k
            for i = 1 : n1
                tmp(i, j) = (M(i, 2 * j - 1) + M(i, 2 * j)) / r2;
                tmp(i, k + j) = (M(i, 2 * j - 1) - M(i, 2 * j)) / r2;
            end
        end
        for j = 1 : 2 * k
            for i = 1 : n1
                M(i, j) = tmp(i, j);
            end
        end
    end
end

function M = haarFWT_2d_inverse(M)
    [n1, n2] = size(M);
    tmp = M;
    r2 = sqrt(2);
    k = 1;
    while(k * 2 <= n2)
        for j = 1 : k
            for i = 1 : n1
                tmp(i, 2 * j - 1) = (M(i, j) + M(i, k + j)) / r2;
                tmp(i, 2 * j) = (M(i, j) - M(i, k + j)) / r2;
            end
        end
        for j = 1 : 2 * k
            for i = 1 : n1
                M(i, j) = tmp(i, j);
            end
        end
        k = k * 2;
    end
    k = 1;
    while(k * 2 <= n1)
        for j = 1 : n2
            for i = 1 : k
                tmp(2 * i - 1, j) = (M(i, j) + M(k + i, j)) / r2;
                tmp(2 * i, j) = (M(i, j) - M(k + i, j)) / r2;
            end
        end
        for j = 1 : n2
            for i = 1 : 2 * k
                M(i, j) = tmp(i, j);
            end
        end
        k = k * 2;
    end
end
