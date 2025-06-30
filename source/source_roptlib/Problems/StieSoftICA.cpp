
#include "Problems/StieSoftICA.h"

/*Define the namespace*/
namespace ROPTLIB{

	StieSoftICA::StieSoftICA(Vector inCs, integer inp)
	{
		Cs = inCs;
        p = inp;
        
        n = inCs.GetElement(0).Getcol();
        N = inCs.Getnumofelements();
        
        NumGradHess = false;
	};

	realdp StieSoftICA::f(const Variable &x) const
	{
        Vector CY(n, p);
        Vector CYs(1, &CY, N);
        CYs.NewMemoryOnWrite();
        
        for(integer i = 0; i < N; i++)
        {
            CYs.GetElement(i).AlphaABaddBetaThis(1, Cs.GetElement(i), GLOBAL::N, x, GLOBAL::N, 0);
        }
        
        Vector D(p * N);
        realdp *Dptr = D.ObtainWriteEntireData();
        const realdp *Yptr = x.ObtainReadData();
        const realdp *CYsptr = CYs.ObtainReadData();
        
        for (integer i = 0; i < N; i++)
        {
            for (integer j = 0; j < p; j++)
            {
                // output Y(:, j)^T CYs(:, j, i),
                Dptr[j + i * p] = dot_(&n, const_cast<realdp *> (Yptr + j * n), &GLOBAL::IONE, const_cast<realdp *> (CYsptr) + n * p * i + j * n, &GLOBAL::IONE);
            }
        }
        
        x.AddToFields("CYs", CYs);
        x.AddToFields("D", D);
        
        return - D.DotProduct(D);
	};

    realdp StieSoftICA::Stof(const Variable &x, const Vector &batch_index) const
    {
        Vector CY(n, p);
        Vector CYs(1, &CY, batch_index.Getlength());
        CYs.NewMemoryOnWrite();
//        batch_index.Print("batch_index:");//---
        for(integer i = 0; i < batch_index.Getlength(); i++)
        {
            CYs.GetElement(i).AlphaABaddBetaThis(1, Cs.GetElement(batch_index[i]), GLOBAL::N, x, GLOBAL::N, 0);
        }
        
        Vector D(p * batch_index.Getlength());
        realdp *Dptr = D.ObtainWriteEntireData();
        const realdp *Yptr = x.ObtainReadData();
        const realdp *CYsptr = CYs.ObtainReadData();
        
        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            for (integer j = 0; j < p; j++)
            {
                // output Y(:, j)^T CYs(:, j, i),
                Dptr[j + i * p] = dot_(&n, const_cast<realdp *> (Yptr + j * n), &GLOBAL::IONE, const_cast<realdp *> (CYsptr) + n * p * i + j * n, &GLOBAL::IONE);
            }
        }
        
        x.AddToFields("CYs", CYs);
        x.AddToFields("D", D);
        
        return - D.DotProduct(D);
    };

    Vector &StieSoftICA::EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const
    {
        Vector CYs = x.Field("CYs"), D = x.Field("D");
        const realdp *CYsptr = CYs.ObtainReadData();
        const realdp *Dptr = D.ObtainReadData();
        result->SetToZeros();
        realdp *resultptr = result->ObtainWritePartialData();
        
        realdp coef = 0;
        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            for (integer j = 0; j < p; j++)
            {
                coef = -Dptr[j + i * p] * 4;
                // result(:, j) <- coef * CYs(:, j, i) + result(:, j)
                axpy_(&n, &coef, const_cast<realdp *> (CYsptr + i * n * p + j * n), &GLOBAL::IONE, resultptr + j * n, &GLOBAL::IONE);
            }
        }
        return *result;
    };

	Vector &StieSoftICA::EucGrad(const Variable &x, Vector *result) const
	{
        Vector CYs = x.Field("CYs"), D = x.Field("D");
        const realdp *CYsptr = CYs.ObtainReadData();
        const realdp *Dptr = D.ObtainReadData();
        result->SetToZeros();
        realdp *resultptr = result->ObtainWritePartialData();
        
        realdp coef = 0;
        for (integer i = 0; i < N; i++)
        {
            for (integer j = 0; j < p; j++)
            {
                coef = -Dptr[j + i * p] * 4;
                // result(:, j) <- coef * CYs(:, j, i) + result(:, j)
                axpy_(&n, &coef, const_cast<realdp *> (CYsptr + i * n * p + j * n), &GLOBAL::IONE, resultptr + j * n, &GLOBAL::IONE);
            }
        }
        return *result;
	};

	Vector &StieSoftICA::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
	{
        const realdp *Yptr = x.ObtainReadData();
        const realdp *etaxptr = etax.ObtainReadData();
        result->SetToZeros();
        realdp *resultptr = result->ObtainWritePartialData();
        Vector CYs = x.Field("CYs"), D = x.Field("D");
        const realdp *CYsptr = CYs.ObtainReadData();
        const realdp *Dptr = D.ObtainReadData();
        
        const realdp *Csptr = Cs.ObtainReadData();
        Vector tmp(n, p);
        realdp *tmpptr = nullptr;
        realdp coef = 0;
        
		for (integer i = 0; i < N; i++)
		{
            tmp = etax;
            tmpptr = tmp.ObtainWritePartialData();
            
			for (integer j = 0; j < p; j++)
			{
				// temp(:, j) <- D(j, i) * temp(:, j)
				scal_(&n, const_cast<realdp *> (Dptr + i * p + j), tmpptr + j * n, &GLOBAL::IONE);
			}
			for (integer j = 0; j < p; j++)
			{
				// output etax(:, j)^T CYs(:, j, i)
				coef = dot_(&n, const_cast<realdp *> (etaxptr + j * n), &GLOBAL::IONE, const_cast<realdp *> (CYsptr + i * n * p + j * n), &GLOBAL::IONE) * 2;
				// tmp(:, j) <- coef * Y(:, j) + tmp(:, j)
				axpy_(&n, &coef, const_cast<realdp *> (Yptr + j * n), &GLOBAL::IONE, tmpptr + j * n, &GLOBAL::IONE);
			}

			// result <- result + Cs(:, :, i) * temp
			gemm_(GLOBAL::N, GLOBAL::N, &n, &p, &n, &GLOBAL::DONE, const_cast<realdp *> (Csptr) + i * n * n, &n, tmpptr, &n, &GLOBAL::DONE, resultptr, &n);
		}

        result->ScalarTimesThis(-4.0);
        
        return *result;
	};
}; /*end of ROPTLIB namespace*/
