% function MTestStieSPCA()
% load('testimage.mat');

    seed = floor(rand() * 100000);
    seed = 20;
    fprintf('CTestFRankETextureInpainting seed:%d\n', seed);
    rng(seed);
    
%     load('flower.mat');
%     A = double(rgb2gray(flower));
    
    A = double(imread('cameraman.tif'));
    normA = norm(A);
    A = A / normA;
    [m, n] = size(A);
    r = 100;
    
    B = rand(m, n);
    A(B>0.5) = 0;
%     S = svd(A);
%     return;
    
    [U, D, V] = svd(A);
    U = U(:, 1:r);
    D = D(1:r, 1:r);
    V = V(:, 1:r);
    
%     U = orth(randn(m, r));
%     D = randn(r, r);
%     V = orth(randn(n, r));
    Xinitial.main = U * D * V';
    Xinitial.U = U;
    Xinitial.D = D;
    Xinitial.V = V;
    type = 1;
    LADMmu = 0.1;
    LADMrho = 1.1;
    LADMeta = 3;
    
    %method: 1: AManPG, 2: LADM
    method = 1; 
    lambda = 0.0001;
    
%     method = 2;
%     lambda = 1;
    
    
    SolverParams.method = 'IARPG'; % IRPG IARPG
    SolverParams.IsCheckParams = 1;
%     SolverParams.RPGVariant = 0; %0: RPG without adaptive stepsize, 1: RPG with adaptive stepsize
    SolverParams.LengthW = 1;
    SolverParams.OutputGap = 1;
    SolverParams.Max_Iteration = 500;
    SolverParams.SMtol = 1e-2;
    SolverParams.Tolerance = 1e-3;
%     SolverParams.Min_Iteration = 10;
    SolverParams.Verbose = 2;
    
%     [xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times] = ...
%         TestFRankETextureInpainting(sparse(A), lambda, Xinitial, type, method, SolverParams);
% m*n
% nnz(sparse(A))
% return;
    [xopt1] = TestFRankETextureInpainting(sparse(A), lambda, Xinitial, type, method, SolverParams, LADMmu, LADMrho, LADMeta);
    
    xopt1.main = xopt1.main * normA;
%     svd(xopt1.main)'
    
    % figure(1);
    % subplot(1, 3, 1);
    % imagesc(A);
    % title('Original image');
    % if(type == 2)
    %     if(method == 1)
    %         subplot(1, 3, 2);
    %         rA = dct2(xopt1.main);
    %         imagesc(rA);
    %         title('Recovered image AManPG');
    %     else
    %         subplot(1, 3, 3);
    %         rA = dct2(xopt1.main);
    %         imagesc(rA);
    %         title('Recovered image LADM');
    %     end
    %     max(max(A - rA))
    %     max(max(A))
    %     here = norm(full(A - rA))
    % else
    %     if(method == 1)
    %         subplot(1, 3, 2);
    %     else
    %         subplot(1, 3, 3);
    %     end
    %     rA = haarFWT_2d_inverse(xopt1.main);
    %     imagesc(rA);
    % end
    % norm(A / norm(A, 'fro') - rA / norm(rA, 'fro'), 'fro')
    
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
