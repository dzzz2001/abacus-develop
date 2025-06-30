rng(1);

fprintf('CTestCFRankQ2FBlindDecon2D\n');
load('testimage.mat')
% testimage = round(rand(8, 8) * 256);
% testimage = testimage((1:512) + 256, (1:512)+256);
Itrue = double(testimage(1:end, 1:end))/255;
n1 = size(Itrue,1);
n2 = size(Itrue,2);
L = n1 * n2;
r = 1;
%% blur kernel and corresponding mask matrix B
kernel = fspecial('motion',50,45);
[k1,k2] = size(kernel);
htrue = zeros(n1,n2);
htrue((n1)/2-(k1+1)/2+2:(n1)/2+(k1+1)/2,(n2+1)/2-(k2+1)/2+2:(n2+1)/2+(k2+1)/2)=kernel;

%     %% Gaussian kernel
%     sigma = [1, 0.5; 0.5, 2];
%     kernel = gaussiankernal(sigma, 11, 11);
%     kernel(find(kernel < 1e-3)) = 0;
%     [k1,k2] = size(kernel);
%     htrue = zeros(n1,n2);
%     htrue(ceil((n1-k1)/2) + 1:ceil((n1-k1)/2) + k1, ceil((n2 - k2)/2) + 1:ceil((n2 - k2)/2) + k2)=kernel;
%     sum(sum(kernel > 0))
%     return;

htrue = fftshift(htrue);
maskB = zeros(n1,n2); % support of kernel
maskB(abs(htrue)>0)=1;
B = spdiags(complex(maskB(:), eps * ones(L, 1)), 0, L, L);

%% Wavelet and corresponding mask matrix C
Iblur = ifft2(fft2(Itrue).*fft2(htrue)); % same as Iblur = imfilter(Itrue,kernel,'circular');
[WIblur] = haarFWT_2d(Iblur);
%     [WIblur] = haarFWT_2d(Itrue);

dimW = round(L / 4); % 5000;% 
[~,ind] = sort(abs(WIblur(:)),'descend');%%%%%% should be mriconW
maskW = zeros(size(WIblur)); %%%%%%%
maskW(ind(1:dimW)) = 1;
C = spdiags(complex(maskW(:), eps * ones(L, 1)), 0, L, L);

%% True solution
[WItrue] = haarFWT_2d(Itrue);
truesoln = zeros(4 * L * r, 1);
truesoln(1:2:end) = [htrue(:); WItrue(:)];

%% Initial iterate
n = sqrt(n1*n2);
AA = @(x) 1/n*fft2(haarFWT_2d_inverse(maskW.*reshape(x,n1,n2)));
AAt = @(Ax) maskW.*(haarFWT_2d(n*ifft2(Ax)));
BB = @(h) 1/n*fft2(maskB.*h);
BBt = @(Bh) maskB.*(n*ifft2(Bh));
y = fft2(Iblur);
%     y = complex(full(y), eps * ones(size(y)));
start = spectral_init(y/L,AA,AAt,BB,BBt,L);
hinit = start.u * sqrt(start.s);
minit = start.v * sqrt(start.s);
xinit = complex([hinit(:); minit(:)]);
% xinit = complex(truesoln(1:2:end), truesoln(2:2:end));%%---

SolverParams.method = 'RSD';
SolverParams.IsCheckParams = 1;
SolverParams.Max_Iteration = 80;
SolverParams.OutputGap = 1;
%     SolverParams.LengthSY = 20;
SolverParams.Verbose = 2;
%     SolverParams.LS_ratio1 = 0.1;
%     SolverParams.LS_ratio2 = 0.9;
%     SolverParams.Num_pre_BB = 0;
%     SolverParams.BBratio = 1;
%     SolverParams.Tolerance = 1e-7;
HasHHR = 0;
rho = start.s^2/100;
mu = (6/log(L))*sqrt(L/(dimW + sum(sum(kernel ~= 0))));%(1000/log(L))*sqrt(L/(dimW+sum(sum(htrue))));
d = 50*norm(WIblur,'fro');
ParamSet = 1;
[Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, dists] = TestCFRankQ2FBlindDecon2D(y(:), B, C, xinit, n1, n2, r, HasHHR, rho, d, mu, ParamSet, SolverParams);

soln = complex(Xopt.main(1:2:end), Xopt.main(2:2:end));
WIfinal = reshape(soln(L + 1 : end), n1, n2);
Ifinal = haarFWT_2d_inverse(maskW.*WIfinal);
RealIfinal = (Ifinal .* conj(Ifinal)).^0.5;

fprintf('Final relres:%f, nBh/nCm:%d, nFFT:%d, relerr:%f\n', f.^0.5/norm(y), nf + ng, 2 * nf + 3 * ng, norm(Itrue - RealIfinal / norm(RealIfinal, 'fro') * norm(Iblur, 'fro'), 'fro') / norm(Itrue, 'fro'));

% figure(1);clf;
% subplot(1, 3, 1);
% imshow(Itrue);
% subplot(1, 3, 2);
% imshow(Iblur);
% subplot(1, 3, 3);
% imshow(RealIfinal / norm(RealIfinal, 'fro') * norm(Iblur, 'fro'));
    

function MTestCFR2BlindDecon2D()
    seed = floor(rand() * 100000);
    seed = 2;
    fprintf('CTestCFRankQ2FBlindDecon2D seed:%d\n', seed);
    rng(seed);
    n1 = 32;
    n2 = 32;
    L = n1 * n2;
    r = 1;
    
%     idx2 = randperm(L);
%     idx2 = idx2(1:L);
    B = spdiags(complex(ones(L, 1), eps * ones(L, 1)), 0, L, L);
%     B = Breal(:,idx2);
    
    C = spdiags(complex(ones(L, 1), eps * ones(L, 1)), 0, L, L);
    y = complex(randn(L, 1), randn(L, 1));
    
    Xinitial = randn(2*(L + L) * r);
    
%     SolverParams.method = 'RSD';
%     SolverParams.method = 'LRBFGS';
%     SolverParams.method = 'RTRSR1';
    SolverParams.method = 'RTRNewton';
    SolverParams.IsCheckParams = 0;
%     SolverParams.IsCheckGradHess = 1;
    SolverParams.Max_Iteration = 100;
    SolverParams.OutputGap = 1;
    SolverParams.LengthSY = 4;
    SolverParams.Verbose = 2;
%     SolverParams.InitSteptype = 1;
    HasHHR = 0;
    rho = 0.;
    d = 1;
    mu = 1;
    ParamSet = 1;
    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime, funs, grads, times, dists] = TestCFRankQ2FBlindDecon2D(y, B, C, Xinitial, n1, n2, r, HasHHR, rho, d, mu, ParamSet, SolverParams);
end

function start = spectral_init(yo,A,At,B,Bt,L)
    v = randn(1,L);
    niter = 50;
    for i = 1 : niter
        u = Bt(yo.*conj(A(conj(v)))); 
        s = norm(u,'fro');
        u = u/s;

        v = conj(At(conj(B(u)).*yo));
        s = norm(v,'fro');
        v = v/s;
        
%         u = Bt(yo.*(A((v)))); 
%         s = norm(u,'fro');
%         u = u/s;
%         v = (At((B(u)).*conj(yo)));
%         s = norm(v,'fro');
%         v = v/s;
    end
    start.u = u;
    start.s = s;
    start.v = conj(v);
end

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

function kernel = gaussiankernal(sigma, xs, ys)
    x = linspace(-2, 2, xs);
    y = linspace(-2, 2, ys);
    [X, Y] = meshgrid(x, y);
    sigma = sigma / det(sigma);
    invsigma = pinv(sigma);
    kernel = exp(-0.5 * (X .* X * invsigma(1, 1) + X .* Y * invsigma(1, 2) + Y .* X * invsigma(2, 1) + Y .* Y * invsigma(2, 2))) / 2 / pi;
    kernel = kernel / sum(sum(kernel));
end
