
#include "Problems/GrassSVPCA.h"

/*Define the namespace*/
namespace ROPTLIB{

	GrassSVPCA::GrassSVPCA(Vector inB, realdp inlambda, integer inn, integer inm, integer inp)
	{
		B = inB;
		n = inn;
        m = inm;
		p = inp;
        lambda = inlambda;
        
        B.SVDDecom();
        Vector S = B.Field("_S");
        L = S.ObtainReadData()[0] * S.ObtainReadData()[0] * 2;
        B.RemoveAllFromFields();
        
        NumGradHess = false;
	};

	GrassSVPCA::~GrassSVPCA(void)
	{
	};

	realdp GrassSVPCA::f(const Variable &x) const
	{
        Vector Bx(m, p); Bx.AlphaABaddBetaThis(1, B, GLOBAL::N, x, GLOBAL::N, 0);/* Bx = B * x; */
        realdp result = - Bx.DotProduct(Bx);
        
        x.AddToFields("Bx", Bx);
//        std::cout << "fpart:" << result << std::endl;//---
        return result;
	};

    realdp GrassSVPCA::g(const Variable &x) const
    {
        realdp result = 0;
        for(integer i = 0; i < n; i++)
        {
            result += lambda * x.GetSubmatrix(i, i, 0, p-1).Fnorm();
        }
//        std::cout << "gpart:" << result << std::endl;//---
        return result;
    };

	Vector &GrassSVPCA::EucGrad(const Variable &x, Vector *result) const
	{
        Vector Bx = x.Field("Bx");
        result->AlphaABaddBetaThis(-2, B, GLOBAL::T, Bx, GLOBAL::N, 0); /* -2.0 * (B.GetTranspose() * Bx) */
        
        return *result;
	};

	Vector &GrassSVPCA::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
	{
        Vector Betax(m, p); Betax.AlphaABaddBetaThis(1, B, GLOBAL::N, etax, GLOBAL::N, 0);
        result->AlphaABaddBetaThis(-2, B, GLOBAL::T, Betax, GLOBAL::N, 0); /* -2.0 * (B.GetTranspose() * (B * etax)) */
        
        return *result;
	};

    Vector &GrassSVPCA::PreConditioner(const Variable &x, const Vector &eta, Vector *result) const
    {
        *result = Vector (1);
        realdp *resultptr = result->ObtainWriteEntireData();
        resultptr[0] = L;
        return *result;
    };

    Vector &GrassSVPCA::ProxW(const Vector &x, const Vector &Weight, Vector *result) const
    { /* (1 - lambda / max(\|X_{i, :}\| * W, lambda)) X_{i, :}*/
        realdp W = Weight.ObtainReadData()[0];
        Vector D(n);
        realdp *Dptr = D.ObtainWriteEntireData();
        for(integer i = 0; i < n; i++)
        {
            Dptr[i] = (1.0 - lambda / std::max( x.GetSubmatrix(i, i, 0, p-1).Fnorm() * W, lambda));
        }
        *result = D.GetDiagTimesM(x);
        
        return *result;
    };

    Vector &GrassSVPCA::CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const
    { /* if \|X_{i, :}\| <= \lambda, result = 0, otherwise, result = (1 - lambda / (W * \|X_{i, :}\|) ) eta_{i, :} + (lambda * X_{i, :} * eta_{i, :} / (W * \|X_{i, :}\|^3) ) * X_{i, :}; */
        realdp W = Weight.ObtainReadData()[0];
        Vector D(n);
        realdp *Dptr = D.ObtainWriteEntireData();
        for(integer i = 0; i < n; i++)
        {
            Dptr[i] = (1.0 - lambda / std::max( x.GetSubmatrix(i, i, 0, p-1).Fnorm() * W, lambda));
        }
        *result = D.GetDiagTimesM(eta);
        
        for(integer i = 0; i < n; i++)
        {
            realdp tmp = x.GetSubmatrix(i, i, 0, p-1).Fnorm();
            if(W * tmp <= lambda)
                Dptr[i] = 0;
            else
                Dptr[i] = lambda * x.GetSubmatrix(i, i, 0, p-1).DotProduct(eta.GetSubmatrix(i, i, 0, p-1)) / (W * tmp * tmp * tmp);
        }
        result->AlphaXaddThis(1, D.GetDiagTimesM(x));
        
        return *result;
    };
}; /*end of ROPTLIB namespace*/
