
#include "Problems/ObliqueSPCA.h"

/*Define the namespace*/
namespace ROPTLIB{
	ObliqueSPCA::ObliqueSPCA(Vector inA, realdp inlambda, integer inn, integer inm, integer inp)
	{
		A = inA;
		n = inn;
        m = inm;
		p = inp;
        lambda = inlambda;
        
        A.SVDDecom();
        Vector S = A.Field("_S");
        L = S.Fnorm(); L = L * L * 8;
//        L = S.Fnorm();
        
        Dsq = Vector(p, p);
        Dsq.SetToZeros();
        realdp *Dsqptr = Dsq.ObtainWriteEntireData();
        const realdp *Sptr = S.ObtainReadData();
        for(integer i = 0; i < p; i++)
        {
            Dsqptr[i + i * p] = Sptr[i];
        }
        A.RemoveAllFromFields();
        
        NumGradHess = false;
	};

	ObliqueSPCA::~ObliqueSPCA(void)
	{
	};

	realdp ObliqueSPCA::f(const Variable &x) const
	{
        Vector Ax(m, p); Ax.AlphaABaddBetaThis(1, A, GLOBAL::N, x, GLOBAL::N, 0);/* Ax = A * x; */
        Vector xtAtAxmDsq(p, p); xtAtAxmDsq.AlphaABaddBetaThis(1, Ax, GLOBAL::T, Ax, GLOBAL::N, 0);
        xtAtAxmDsq.AlphaXaddThis(-1, Dsq);
        realdp result = xtAtAxmDsq.DotProduct(xtAtAxmDsq);
        x.AddToFields("Ax", Ax);
        x.AddToFields("xtAtAxmDsq", xtAtAxmDsq);
        return result;
	};

    realdp ObliqueSPCA::g(const Variable &x) const
    {
        const realdp *xptr = x.ObtainReadData();
        realdp result = 0;
        for(integer i = 0; i < n * p; i++)
            result += lambda * std::fabs(xptr[i]);
        return result;
    };

	Vector &ObliqueSPCA::EucGrad(const Variable &x, Vector *result) const
	{
        Vector Ax = x.Field("Ax");
        Vector xtAtAxmDsq = x.Field("xtAtAxmDsq");
        Vector tmp(m, p); tmp.AlphaABaddBetaThis(1, Ax, GLOBAL::N, xtAtAxmDsq, GLOBAL::N, 0);
        result->AlphaABaddBetaThis(4, A, GLOBAL::T, tmp, GLOBAL::N, 0);
        
        return *result;
	};

	Vector &ObliqueSPCA::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
	{
        Vector Ax = x.Field("Ax");
        Vector xtAtAxmDsq = x.Field("xtAtAxmDsq");
        Vector Aetax = A * etax;
        *result = 4 * A.GetTranspose() * ( (Ax * ( Aetax.GetTranspose() * Ax + Ax.GetTranspose() * Aetax ) ) + Aetax * xtAtAxmDsq );
        
        return *result;
	};

    Vector &ObliqueSPCA::PreConditioner(const Variable &x, const Vector &eta, Vector *result) const
    {
        *result = Vector (1);
        realdp *resultptr = result->ObtainWriteEntireData();
        resultptr[0] = L;
        return *result;
    };

    Vector &ObliqueSPCA::ProxW(const Vector &x, const Vector &Weight, Vector *result) const
    { /* output = min(0, X + mu ./ W) + max(0, X - mu ./ W) */
        realdp *resultptr = result->ObtainWriteEntireData();
        const realdp *xptr = x.ObtainReadData();
        const realdp *Wptr = Weight.ObtainReadData();
        
        integer nblock = Weight.Getlength();
        integer blocksize = x.Getlength() / nblock;
        integer idx = 0;
        
        for(integer i = 0; i < nblock; i++)
        {
            for(integer j = 0; j < blocksize; j++)
            {
                idx = j + i * blocksize;
                resultptr[idx] = ((xptr[idx] + lambda / Wptr[i] < 0.0) ? xptr[idx] + lambda / Wptr[i] : 0.0) + ((xptr[idx] - lambda / Wptr[i] > 0.0) ? xptr[idx] - lambda / Wptr[i] : 0.0);
            }
        }
        return *result;
    };

    Vector &ObliqueSPCA::CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const
    { /* output = (abs(x) .* W > lambda) .* eta; */
        const realdp *xptr = x.ObtainReadData();
        const realdp *etaptr = eta.ObtainReadData();
        realdp *resultptr = result->ObtainWriteEntireData();
        const realdp *Wptr = Weight.ObtainReadData();
        
        integer nblock = Weight.Getlength();
        integer blocksize = x.Getlength() / nblock;
        integer idx = 0;
        
        for(integer i = 0; i < nblock; i++)
        {
            for(integer j = 0; j < blocksize; j++)
            {
                idx = j + i * blocksize;
                resultptr[idx] = (std::abs(xptr[idx]) * Wptr[i] > lambda) ? etaptr[idx] : 0;
            }
        }
        return *result;
    };
}; /*end of ROPTLIB namespace*/
