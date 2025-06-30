
#include "Problems/FRankE3FMatCompletion.h"

/*Define the namespace*/
namespace ROPTLIB{

	FRankE3FMatCompletion::FRankE3FMatCompletion(unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdp *inV, integer innz, integer inm, integer inn, integer inr)
	{
		ir = inir;
		jc = injc;
        jcc = injcc;
		vals = inV;
		nz = innz;
		m = inm;
		n = inn;
		r = inr;
        NumGradHess = false;
//        SMEucRepGrad = nullptr;
//        SMEucRepHess = nullptr;
	};

	FRankE3FMatCompletion::~FRankE3FMatCompletion(void)
	{
//        if(SMEucRepGrad != nullptr)
//            delete SMEucRepGrad;
//        if(SMEucRepHess != nullptr)
//            delete SMEucRepHess;
	};

	void FRankE3FMatCompletion::ProjecOmegaUDVT(const realdp *U, const realdp *D, const realdp *V, integer inm, integer inn, integer inr, unsigned long *inir, unsigned long *injc, unsigned long *injcc, integer nz, realdp *result)
	{
		realdp *DtUt = new realdp[inm * inr + inn * inr];
        realdp *Vt = DtUt + inm * inr;
        for(integer i = 0; i < inn; i++)
        {
            for(integer j = 0; j < inr; j++)
            {
                Vt[j + i * inr] = V[i + j * inn];
            }
        }
        
		gemm_(GLOBAL::T, GLOBAL::T, &inr, &inm, &inr, &GLOBAL::DONE, const_cast<realdp *> (D), &inr, const_cast<realdp *> (U), &inm, &GLOBAL::DZERO, DtUt, &inr);

       for (integer i = 0; i < nz; i++)
       {
           result[i] = dot_(&inr, const_cast<realdp *> (DtUt) + inir[i] * inr, &GLOBAL::IONE, const_cast<realdp *> (Vt) + injc[i] * inr, &GLOBAL::IONE);
       }

		delete[] DtUt;
        
        
// 		realdp *UD = new realdp[inm * inr];
// 		gemm_(GLOBAL::N, GLOBAL::N, &inm, &inr, &inr, &GLOBAL::DONE, const_cast<realdp *> (U), &inm, const_cast<realdp *> (D), &inr, &GLOBAL::DZERO, UD, &inm);
//        
// //        for (integer i = 0; i < nz; i++)
// //        {
// //            result[i] = dot_(&inr, const_cast<realdp *> (UD) + inir[i], &inm, const_cast<realdp *> (V) + injc[i], &inn);
// //        }
//        
// 		for (integer i = 0; i < nz; i++)
// 		{
// 			result[i] = 0;
// 			for (integer j = 0; j < inr; j++)
// 			{
// 				/*row: inir[i], col: injc[i]*/
// 				result[i] += UD[static_cast<integer> (inir[i]) + j * inm] * V[static_cast<integer> (injc[i]) + j * inn];
// 			}
// 		}
// 		delete[] UD;
	};

	realdp FRankE3FMatCompletion::f(const Variable &x) const
	{
        const realdp *Uptr = x.GetElement(0).ObtainReadData();
        const realdp *Dptr = x.GetElement(1).ObtainReadData();
        const realdp *Vptr = x.GetElement(2).ObtainReadData();
        
//        Vector UDVtmA(nz);
//        realdp *UDVtmAptr = UDVtmA.ObtainWriteEntireData();
        realdp *UDVtmAptr = new realdp[nz];
        unsigned long *iir = new unsigned long[nz], *jjc = new unsigned long[nz], *jjcc = new unsigned long[n + 1];

        ProjecOmegaUDVT(Uptr, Dptr, Vptr, m, n, r, ir, jc, jcc, nz, UDVtmAptr);
        
        /*P_Omage(U D V^T - A)*/
        axpy_(const_cast<integer *> (&nz), &GLOBAL::DNONE, vals, &GLOBAL::IONE, UDVtmAptr, &GLOBAL::IONE);
        
        realdp result = 0.5 * dot_(&nz, UDVtmAptr, &GLOBAL::IONE, UDVtmAptr, &GLOBAL::IONE);
        
        for(integer i = 0; i < nz; i++)
        {
            iir[i] = ir[i];
            jjc[i] = jc[i];
        }
        
        for(integer i = 0; i < n + 1; i++)
        {
            jjcc[i] = jcc[i];
        }
        x.AddSparseMatrixToFields(new SparseMatrix(m, n, iir, jjc, jjcc, UDVtmAptr, nz));
        
//        x.AddToFields("UDVtmA", UDVtmA);
//        if(SMEucRepGrad != nullptr)
//            delete SMEucRepGrad;
//
//        SMEucRepGrad = new SparseMatrix(m, n, ir, jc, UDVtmAptr, nz);
//        Vector EucRepinX;
//        EucRepinX.SetSharedSparseMatrix(SMEucRepGrad);
//
//        x.AddToFields("EucRepinX", EucRepinX);
        
        return result;
	};
	
	Vector &FRankE3FMatCompletion::EucGrad(const Variable &x, Vector *result) const
	{
        Vector EucRepinX = x.Field("_SparseMatrix");
        result->AddToFields("_SparseMatrix", EucRepinX);
//        result->Print("result:", false);//---
        return *result;
	};
	
	Vector &FRankE3FMatCompletion::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
	{
//        Vector EucRep(nz);
//        realdp *EucRepptr = EucRep.ObtainWriteEntireData();
        realdp *EucRepptr = new realdp[nz];
        unsigned long *iir = new unsigned long[nz], *jjc = new unsigned long[nz], *jjcc = new unsigned long[n + 1];
        
        const realdp *Uptr = x.GetElement(0).ObtainReadData();
        const realdp *Dptr = x.GetElement(1).ObtainReadData();
        const realdp *Vptr = x.GetElement(2).ObtainReadData();
        
        const realdp *dUptr = etax.GetElement(0).ObtainReadData();
        const realdp *dDptr = etax.GetElement(1).ObtainReadData();
        const realdp *dVptr = etax.GetElement(2).ObtainReadData();
        
        ProjecOmegaUDVT(dUptr, Dptr, Vptr, m, n, r, ir, jc, jcc, nz, EucRepptr);
        realdp *tmp = new realdp[nz];
        ProjecOmegaUDVT(Uptr, dDptr, Vptr, m, n, r, ir, jc, jcc, nz, tmp);
        axpy_(&nz, &GLOBAL::DONE, tmp, &GLOBAL::IONE, EucRepptr, &GLOBAL::IONE);
        ProjecOmegaUDVT(Uptr, Dptr, dVptr, m, n, r, ir, jc, jcc, nz, tmp);
        axpy_(&nz, &GLOBAL::DONE, tmp, &GLOBAL::IONE, EucRepptr, &GLOBAL::IONE);
        delete[] tmp;
        
        for(integer i = 0; i < nz; i++)
        {
            iir[i] = ir[i];
            jjc[i] = jc[i];
        }
        for(integer i = 0; i < n + 1; i++)
            jjcc[i] = jcc[i];
        result->AddSparseMatrixToFields(new SparseMatrix(m, n, iir, jjc, jjcc, EucRepptr, nz));

//
//        if(SMEucRepHess != nullptr)
//            delete SMEucRepHess;
//
//        SMEucRepHess = new SparseMatrix(m, n, ir, jc, EucRepptr, nz);
//
//        Vector SMEucRep;
//        SMEucRep.SetSharedSparseMatrix(SMEucRepHess);
//
//        result->AddToFields("SparseEucRep", SMEucRep);
        
        return *result;
	};
}; /*end of ROPTLIB namespace*/
