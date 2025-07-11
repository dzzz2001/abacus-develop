#include <assert.h>
#include "diag_cusolver.cuh"
#include "helper_cuda.h"

Diag_Cusolver_gvd::Diag_Cusolver_gvd(cudaStream_t stream_in){
// step 1: create cusolver/cublas handle
    stream = stream_in;
    cusolverH = NULL;
    checkCudaErrors( cusolverDnCreate(&cusolverH) );
    cusolverDnSetStream(cusolverH, stream);

    itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    uplo = CUBLAS_FILL_MODE_LOWER;

    d_A = NULL;
    d_B = NULL;
    d_work = NULL;

    d_A2 = NULL;
    d_B2 = NULL;
    d_work2 = NULL;

    d_W = NULL;
    devInfo = NULL;

    lwork = 0;
    info_gpu = 0;
    is_init = 0;
}

void Diag_Cusolver_gvd::finalize(){
// free resources and destroy
    if (d_A      ) {checkCudaErrors( cudaFree(d_A) );  d_A  = NULL;}
    if (d_B      ) {checkCudaErrors( cudaFree(d_B) );  d_B  = NULL;}
    if (d_A2     ) {checkCudaErrors( cudaFree(d_A2) ); d_A2 = NULL;}
    if (d_B2     ) {checkCudaErrors( cudaFree(d_B2) ); d_B2 = NULL;}
    if (d_W      ) {checkCudaErrors( cudaFree(d_W) );  d_W  = NULL;}
    if (devInfo  ) {checkCudaErrors( cudaFree(devInfo) );   devInfo = NULL;}
}

Diag_Cusolver_gvd::~Diag_Cusolver_gvd(){
    finalize();
    if (cusolverH) {checkCudaErrors( cusolverDnDestroy(cusolverH) );    cusolverH = NULL;}
    //checkCudaErrors( cudaDeviceReset() );
}


void Diag_Cusolver_gvd::init_double(int N){
// step 2: Malloc A and B on device
    m = lda = N;
    checkCudaErrors( cudaMalloc ((void**)&d_A, sizeof(double) * lda * m) );
    checkCudaErrors( cudaMalloc ((void**)&d_B, sizeof(double) * lda * m) );
    checkCudaErrors( cudaMalloc ((void**)&d_W, sizeof(double) * m) );
    checkCudaErrors( cudaMalloc ((void**)&devInfo, sizeof(int)) );
}

void Diag_Cusolver_gvd::init_complex(int N){
// step 2: Malloc A and B on device
    m = lda = N;
    checkCudaErrors( cudaMalloc ((void**)&d_A2, sizeof(cuDoubleComplex) * lda * m) );
    checkCudaErrors( cudaMalloc ((void**)&d_B2, sizeof(cuDoubleComplex) * lda * m) ); 
    checkCudaErrors( cudaMalloc ((void**)&d_W, sizeof(double) * m) );
    checkCudaErrors( cudaMalloc ((void**)&devInfo, sizeof(int)) );
}
        
void Diag_Cusolver_gvd::Dngvd_double(int N, int M, double *A, double *B, double *W, double *V){

    // copy A, B to the GPU
        assert(N == M);
        if (M != m) {
            this->finalize();
            this->init_double(M);
        }
        checkCudaErrors( cudaMemcpyAsync(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream) );
        checkCudaErrors( cudaMemcpyAsync(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream) );

    // Query working space of sygvd
    // The helper functions below can calculate the sizes needed for pre-allocated buffer.
    // The S and D data types are real valued single and double precision, respectively.
    // The C and Z data types are complex valued single and double precision, respectively.
        checkCudaErrors(cusolverDnDsygvd_bufferSize(
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_B,
            lda,
            d_W,
            &lwork
        ));
        checkCudaErrors( cudaMalloc((void**)&d_work, sizeof(double)*lwork) );

    // compute spectrum of (A,B)
        checkCudaErrors(cusolverDnDsygvd(
            cusolverH,
            itype,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_B,
            lda,
            d_W,
            d_work,
            lwork,
            devInfo
        ));

    // copy (W, V) to the cpu root
        checkCudaErrors( cudaMemcpyAsync(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaMemcpyAsync(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaStreamSynchronize(stream) );
        assert(0 == info_gpu);
    // free the buffer
        if (d_work ) checkCudaErrors( cudaFree(d_work) );

}


void Diag_Cusolver_gvd::Dngvd_complex(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V){
    
    // copy A, B to the GPU
        assert(N == M);
        if (M != m) {
            this->finalize();
            this->init_complex(M);
        }
        checkCudaErrors( cudaMemcpyAsync(d_A2, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice, stream) );
        checkCudaErrors( cudaMemcpyAsync(d_B2, B, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice, stream) );

    // Query working space of Zhegvd
    // The helper functions below can calculate the sizes needed for pre-allocated buffer.
    // The S and D data types are real valued single and double precision, respectively.
    // The C and Z data types are complex valued single and double precision, respectively.
        checkCudaErrors( 
            cusolverDnZhegvd_bufferSize(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A2,
                lda,
                d_B2,
                lda,
                d_W,
                &lwork)
        );      
        checkCudaErrors( cudaMalloc((void**)&d_work2, sizeof(cuDoubleComplex)*lwork) );

    // compute spectrum of (A,B)
        checkCudaErrors(
            cusolverDnZhegvd(
                cusolverH,
                itype,
                jobz,
                uplo,
                m,
                d_A2,
                lda,
                d_B2,
                lda,
                d_W,
                d_work2,
                lwork,
                devInfo)
        );
        
    // copy (W, V) to the cpu root
        checkCudaErrors( cudaMemcpyAsync(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaMemcpyAsync(V, d_A2, sizeof(std::complex<double>)*lda*m, cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaStreamSynchronize(stream) );
        assert(0 == info_gpu);

    // free the buffer
        if (d_work2 ) checkCudaErrors( cudaFree(d_work2) );
}
