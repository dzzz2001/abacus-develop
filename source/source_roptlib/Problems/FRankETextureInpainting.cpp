
#include "Problems/FRankETextureInpainting.h"

#ifdef ROPTLIB_WITH_FFTW

/*Define the namespace*/
namespace ROPTLIB{

	FRankETextureInpainting::FRankETextureInpainting(unsigned long *inir, unsigned long *injc, unsigned long *injcc, unsigned long innzmax, Vector inA, realdp inlambda, integer inm, integer inn, integer inr, integer inlengthW, integer intype)
	{
        ir = inir;
        jc = injc;
        jcc = injcc;
        nzmax = innzmax;
		A = inA;
        lambda = inlambda;
		m = inm;
		n = inn;
		r = inr;
        lengthW = inlengthW;
        L = 2;
        type = intype;
        
        NumGradHess = false;
	};

	FRankETextureInpainting::~FRankETextureInpainting(void)
	{
	};

	realdp FRankETextureInpainting::f(const Variable &x) const
	{ /* \|P_Omega (invH(X) - A)\|_F^2 + lambda \|X\|_1 */
//    printf("time1:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        Vector invHx;
        if(type == 1)
        {
            invHx = x.GetInvHaarFWT(); /* haar wavelet */
//            std::cout << x.Fnorm() << ":" << invHx.Fnorm() << ":" << invHx.GetHaarFWT().Fnorm() << ":" << invHx.Fnorm() / x.Fnorm() << ":" << invHx.GetHaarFWT().Fnorm() / invHx.Fnorm() << ":" << invHx.GetHaarFWT().Fnorm() / x.Fnorm() << std::endl;//---
        } else
        {
            invHx = x.GetDCST2D(FFTW_REDFT10); /* DCT wavelet */
//            std::cout << x.Fnorm() << ":" << invHx.Fnorm() << ":" << invHx.GetDCST2D(FFTW_REDFT01).Fnorm() << ":" << invHx.Fnorm() / x.Fnorm() << ":" << invHx.GetDCST2D(FFTW_REDFT01).Fnorm() / invHx.Fnorm() << ":" << invHx.GetDCST2D(FFTW_REDFT01).Fnorm() / x.Fnorm() << std::endl;//---
        }
//        printf("time2:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        x.Print("x:");//--
//        invHx.Print("DCT x:");//---
//        A.Print("input DCT A:");//---
        Vector invHxmA(invHx); invHxmA.AlphaXaddThis(-1, A);
//        invHxmA.Print("DCT x - DCT A:");//---
        Vector PinvHxmA(invHxmA); PinvHxmA.SetToZeros();
        realdp *PinvHxmAptr = PinvHxmA.ObtainWritePartialData();
//        const realdp *Aptr = A.ObtainReadData();
        const realdp *invHxmAptr = invHxmA.ObtainReadData();
        for(integer i = 0; i < nzmax; i++)
        {
            PinvHxmAptr[ir[i] + jc[i] * m] = invHxmAptr[ir[i] + jc[i] * m]; // - Aptr[ir[i] + jc[i] * m];
        }
//        printf("time3:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        PinvHxmA.Print("P (DCT x - DCT A):");//---
        realdp result = PinvHxmA.DotProduct(PinvHxmA);
        
//        std::cout << "f first part:" << result << std::endl;//----
        
        x.AddToFields("PinvHxmA", PinvHxmA);
        
//        printf("time4:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        return result;
	};
	
    realdp FRankETextureInpainting::g(const Variable &x) const
    {
        realdp result = 0;
        const realdp *xptr = x.ObtainReadData();
        for(integer i = 0; i < n * m; i++)
            result += lambda * std::fabs(xptr[i]);
        
        return result;
    };

	Vector &FRankETextureInpainting::EucGrad(const Variable &x, Vector *result) const
	{ /* gradient of the smooth term: 2 H(P_Omega(invH(X) - A)) */
//        printf("time5:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        if(type == 1)
        {
            *result = x.Field("PinvHxmA").GetHaarFWT(); result->ScalarTimesThis(2);
        } else
        {
            *result = x.Field("PinvHxmA").GetDCST2D(FFTW_REDFT01); result->ScalarTimesThis(2);
        }
//        printf("time6:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        return *result;
	};
	
	Vector &FRankETextureInpainting::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
	{
        *result = etax; result->ScalarTimesThis(2);
        return *result;
	};

    Vector &FRankETextureInpainting::PreConditioner(const Variable &x, const Vector &eta, Vector *result) const
    {
        Vector exresult(1);
        realdp *exresultptr = exresult.ObtainWriteEntireData();
        exresultptr[0] = L;
        *result = exresult;
        return *result;
    };

    Vector &FRankETextureInpainting::ProxW(const Vector &x, const Vector &Weight, Vector *result) const
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

    Vector &FRankETextureInpainting::CalJW(const Vector &x, const Vector &eta, const Vector &Weight, Vector *result) const
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

#endif
