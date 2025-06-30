function output = SparseMatrixMultiplication(X, tranX, Y, tranY)
    % The sparse matrix multiplication in Matlab is more efficient. Therefore, if ROPTLIB is invoked from Matlab, 
    % then the sparse matrix multiplication in this file is used.
    if(tranX == 0 && tranY == 0)
        output = X * Y;
    elseif(tranX == 0 && tranY == 1)
        output = X * Y';
    elseif(tranX == 1 && tranY == 0)
        output = X' * Y;
    elseif(tranX == 1 && tranY == 1)
        output = X' * Y';
    end
    clear 'X' 'Y' 'tranX' 'tranY'
end

