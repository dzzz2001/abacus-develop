/*
This file defines the class of sparse matrix. It uses the sparse BLAS
to do matrix operations:
math.nist.gov/spblas/

SparseMatrix

---- WH
*/

#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "Others/randgen.h"
#include <cstdarg>
#include <map>
#include <string>
#include "Others/SparseBLAS/blas_sparse.h"
#include "Others/def.h"
#include <sstream>

/*Define the namespace*/
namespace ROPTLIB{

    class SparseMatrix {
    public:
        
        /*construct real sparse matrix. ir and jc are arrays of indices of nonzero entries.
        vals is an array of nonzero values. nz is the number of nonzero entries.
        i.e., vals[i] stores ir[i]-th row, jc[i]-th column entry
        jcc is the compact form of jc.
        i.e., for any k in [jc[i] to jc[i + 1] - 1], the k-th row, i-th column entry is in in vals[]. jcc uses the same format as that in Matlab.
        Note that the memory of ir, jc, jcc, vals will be maintained by this object. In order words, the input
        memory should not be released, and this object will released it automatically.
        If users need to maintain the memory, then invoke Setnullptr(void) after using the object.*/
        SparseMatrix(integer inm, integer inn, unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdp *invals, integer innz);
        
        /*construct complex sparse matrix. ir and jc are arrays of indices of nonzero entries.
        vals is an array of nonzero values. nz is the number of nonzero entries.
        i.e., vals[i] stores ir[i]-th row, jc[i]-th column entry
        jcc is the compact form of jc.
        i.e., for any k in [jc[i] to jc[i + 1] - 1], the k-th row, i-th column entry is in in vals[]. jcc uses the same format as that in Matlab.
        Note that the memory of ir, jc, jcc, vals will be maintained by this object. In order words, the input
        memory should not be released, and this object will released it automatically.
        If users need to maintain the memory, then invoke Setnullptr(void) after using the object.*/
        SparseMatrix(integer inm, integer inn, unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdpcomplex *invals, integer innz);
        
        void Print(const char *name = "") const;
        
//        inline blas_sparse_matrix GetSparseM(void) const { return SparseM; };
        
        inline bool Getiscomplex(void) const { return iscomplex; };
        
        inline integer Getrow(void) const { return m; };
        
        inline integer Getcol(void) const { return n; };
        
        inline integer Getnz(void) const { return nz; };
        
        inline const unsigned long *Getir(void) const { return ir; };
        
        inline const unsigned long *Getjc(void) const { return jc; };
        
        inline const unsigned long *Getjcc(void) const { return jcc; };
        
        inline const void *GetVals(void) const { return iscomplex ? (void *) valscomplex : (void *) valsreal; };
        
        /*if the memory by ir, jc, vals is not deleted automatically, then this function is invoked.*/
        void Setnullptr(void) { ir = nullptr; jc = nullptr;  jcc = nullptr; valsreal = nullptr; valscomplex = nullptr; };
        
        ~SparseMatrix(void);
        
    protected:
        integer m; /*number of rows*/
        integer n; /*number of columns*/
        
        integer nz;
        unsigned long  *ir;
        unsigned long  *jc;
        unsigned long *jcc;
        realdp *valsreal;
        realdpcomplex *valscomplex;
        
//        blas_sparse_matrix SparseM;
        bool iscomplex;
    };
}; /*end of ROPTLIB namespace*/

#endif
