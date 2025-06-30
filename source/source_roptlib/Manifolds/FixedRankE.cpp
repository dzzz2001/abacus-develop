
#include "Manifolds/FixedRankE.h"

/*Define the namespace*/
namespace ROPTLIB{

	FixedRankE::FixedRankE(integer inm, integer inn, integer inr)
	{
		m = inm;
		n = inn;
		r = inr;
		name.assign("FixedRankEmbedded");
        IsIntrApproach = false;
//        IsVectTranSmooth = true;
        ExtrinsicDim = n * m;
        IntrinsicDim = (m - r) * r + r * r + (n - r) * r;
        EMPTYEXTR = Vector(m, n);
        Vector dU = Vector(m, r), dD = Vector (r, r), dV = Vector(n, r);
        EMPTYEXTR.AddToFields("dU", dU);
        EMPTYEXTR.AddToFields("dD", dD);
        EMPTYEXTR.AddToFields("dV", dV);
        EMPTYINTR = Vector(IntrinsicDim);
        EMPTYNORINTR = Vector(ExtrinsicDim - IntrinsicDim);
	};

	FixedRankE::~FixedRankE()
	{
	};

	realdp FixedRankE::Metric(const Variable &x, const Vector &etax, const Vector &xix) const
	{
        Vector U = x.Field("U"), D = x.Field("D"), V = x.Field("V");
        Vector dUeta = etax.Field("dU"), dDeta = etax.Field("dD"), dVeta = etax.Field("dV");
        Vector dUxix = xix.Field("dU"), dDxix = xix.Field("dD"), dVxix = xix.Field("dV");
        Vector dUetaD = dUeta * D, dUxixD = dUxix * D;
        Vector dVetaD = dVeta * D.GetTranspose(), dVxixD = dVxix * D.GetTranspose();
        return dUetaD.DotProduct(dUxixD) + dVetaD.DotProduct(dVxixD) + dDeta.DotProduct(dDxix);
	};

    Variable FixedRankE::RandominManifold(void) const
    {
        Vector U(m, r);
        U.RandGaussian();
        U = U.GetOrth();
        Vector V(n, r);
        V.RandGaussian();
        V = V.GetOrth();
        Vector D(r, r);
        D.RandGaussian();
        
        Vector result = U * D * V.GetTranspose();
        result.AddToFields("U", U);
        result.AddToFields("D", D);
        result.AddToFields("V", V);
        
        return result;
    };

    Vector &FixedRankE::LinearOPEEta(const Variable &x, const LinearOPE &Hx, const Vector &etax, Vector *result) const
    {
        Manifold::LinearOPEEta(x, Hx, etax, result);
        return ExtrProjection(x, *result, result);
    };

    Vector& FixedRankE::ScalarTimesVector(const Variable &x, const realdp &scalar, const Vector &etax, Vector *result) const
    {
        /* when etax and result are the same, the code "result->ScalarTimesThis(scalar)" can destroy dU, dD, and dV in etax.
         Therefore, we store dU, dD, and dV first*/
        Vector dU = etax.Field("dU"), dD = etax.Field("dD"), dV = etax.Field("dV");
        *result = etax; result->ScalarTimesThis(scalar); /* result = scalar * etax; */
        result->AddToFields("dU", scalar * dU);
        result->AddToFields("dD", scalar * dD);
        result->AddToFields("dV", scalar * dV);
        return *result;
    };

    Vector &FixedRankE::ScalarVectorAddVector(const Variable &x, const realdp &scalar, const Vector &etax, const Vector &xix, Vector *result) const
    {
        /* when xix and result are the same, the code "result->ScalarTimesThis(scalar)" can destroy dU, dD, and dV in xix.
         Therefore, we store dU, dD, and dV first*/
        Vector dU = xix.Field("dU"), dD = xix.Field("dD"), dV = xix.Field("dV");
        *result = xix; result->AlphaXaddThis(scalar, etax);
        dU.AlphaXaddThis(scalar , etax.Field("dU"));
        result->AddToFields("dU", dU);
        dD.AlphaXaddThis(scalar, etax.Field("dD"));
        result->AddToFields("dD", dD);
        dV.AlphaXaddThis(scalar, etax.Field("dV"));
        result->AddToFields("dV", dV);
        
        return *result;
    };

    Vector &FixedRankE::VectorLinearCombination(const Variable &x, realdp scalar1, const Vector &etax, realdp scalar2, const Vector &xix, Vector *result) const
    {
        /* when xix and result are the same, the code "result->ScalarTimesThis(scalar)" can destroy dU, dD, and dV in xix.
         Therefore, we store dU, dD, and dV first*/
        Vector inetax = etax, inxix = xix;
        if(! inetax.FieldsExist("dU"))
        {
            ExtrProjection(x, inetax, &inetax);
        }
        if(! inxix.FieldsExist("dU"))
        {
            ExtrProjection(x, inxix, &inxix);
        }
        
        Vector dU = inxix.Field("dU"), dD = inxix.Field("dD"), dV = inxix.Field("dV");
        *result = inxix; result->ScalarTimesThis(scalar2); result->AlphaXaddThis(scalar1, inetax);
        dU.ScalarTimesThis(scalar2); dU.AlphaXaddThis(scalar1, inetax.Field("dU"));
        result->AddToFields("dU", dU);
        dD.ScalarTimesThis(scalar2); dD.AlphaXaddThis(scalar1, inetax.Field("dD"));
        result->AddToFields("dD", dD);
        dV.ScalarTimesThis(scalar2); dV.AlphaXaddThis(scalar1, inetax.Field("dV"));
        result->AddToFields("dV", dV);
        return *result;
    };

    Vector &FixedRankE::ObtainNorVerIntr(const Variable &x, const Vector &etax, Vector *result) const
    {
        Vector U = x.Field("U"), D = x.Field("D"), V = x.Field("V");
        Vector QUTetax = etax.HHRMtp(U.Field("_HHR"), U.Field("_tau"), GLOBAL::T, GLOBAL::L);
        Vector QUTetaxQV = QUTetax.HHRMtp(V.Field("_HHR"), V.Field("_tau"), GLOBAL::N, GLOBAL::R);
        const realdp *QUTetaxQVptr = QUTetaxQV.ObtainReadData();
        realdp *resultptr = result->ObtainWriteEntireData();
        
        for(integer i = 0; i < n - r; i++)
        {
            for(integer j = 0; j < m - r; j++)
            {
                resultptr[j + i * (m - r)] = QUTetaxQVptr[j + r + (i + r) * m];
            }
        }
        return *result;
    };

    Vector &FixedRankE::ObtainNorVerExtr(const Variable &x, const Vector &intretax, Vector *result) const
    {
        const realdp *intretaxptr = intretax.ObtainReadData();
        realdp *resultptr = result->ObtainWriteEntireData();
        for(integer i = 0; i < n; i++)
        {
            for(integer j = 0; j < m; j++)
            {
                if(i < r || j < r)
                    resultptr[j + i * m] = 0;
                else
                    resultptr[j + i * m] = intretaxptr[j - r + (i - r) * (m - r)];
            }
        }
        Vector U = x.Field("U"), D = x.Field("D"), V = x.Field("V");
        if(!(U.FieldsExist("_HHR")))
        {
            U.HHRDecom();
            x.AddToFields("U", U);
        }
        if(!(V.FieldsExist("_HHR")))
        {
            V.HHRDecom();
            x.AddToFields("V", V);
        }
        
        *result = result->HHRMtp(U.Field("_HHR"), U.Field("_tau"), GLOBAL::N, GLOBAL::L);
        *result = result->HHRMtp(V.Field("_HHR"), V.Field("_tau"), GLOBAL::T, GLOBAL::R);

        return *result;
    };

	Variable &FixedRankE::Retraction(const Variable &x, const Vector &etax, Variable *result) const
	{
        Vector Ux = x.Field("U"), Dx = x.Field("D"), Vx = x.Field("V");
        Vector dU = etax.Field("dU"), dD = etax.Field("dD"), dV = etax.Field("dV");
        
        Vector UdU = Ux + dU;
        UdU.QRDecom();
        Vector Q = UdU.Field("_Q"), R = UdU.Field("_R");
        R.SVDDecom();
        Vector Uy = Q * (R.Field("_U") * R.Field("_Vt"));
        Uy.AddToFields("_Vt", R.Field("_Vt"));
        Uy.AddToFields("_S", R.Field("_S"));
        Uy.AddToFields("_HHR", UdU.Field("_HHR"));
        Uy.AddToFields("_tau", UdU.Field("_tau"));
        
        Vector VdV = Vx + dV;
        VdV.QRDecom();
        Q = VdV.Field("_Q"); R = VdV.Field("_R");
        R.SVDDecom();
        Vector Vy = Q * (R.Field("_U") * R.Field("_Vt"));
        Vy.AddToFields("_Vt", R.Field("_Vt"));
        Vy.AddToFields("_S", R.Field("_S"));
        Vy.AddToFields("_HHR", VdV.Field("_HHR"));
        Vy.AddToFields("_tau", VdV.Field("_tau"));
        
        Vector Dy = Dx + dD;
        *result = Uy * Dy * Vy.GetTranspose();
        result->AddToFields("U", Uy);
        result->AddToFields("D", Dy);
        result->AddToFields("V", Vy);
        return *result;
	};

    Vector &FixedRankE::VecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result, bool IsEtaXiSameDir) const
    {
        if(IsEtaXiSameDir)
        {
            Vector Ux = x.Field("U"), dUeta = etax.Field("dU"), Uy = y.Field("U"), dUxix = xix.Field("dU");
            Vector Dx = x.Field("D"), Dy = y.Field("D");
            Vector Vx = x.Field("V"), dVeta = etax.Field("dV"), Vy = y.Field("V"), dVxix = xix.Field("dV");
            
            realdp alpha1 = sqrt(dUxix.DotProduct(dUxix) / dUeta.DotProduct(dUeta));
            Vector Uy_Vt = Uy.Field("_Vt"), Uy_S = Uy.Field("_S");
            Vector tmp1 = dUeta * Uy_Vt.GetTranspose();
            Uy_S = 1 / Uy_S;
            tmp1 = Uy_S.GetDiagTimesM(tmp1, GLOBAL::R) * Uy_Vt;
            Vector result1 = (tmp1 - Uy * (tmp1.GetTranspose() * tmp1)) * alpha1;
            
            realdp alpha2 = sqrt(dVxix.DotProduct(dVxix) / dVeta.DotProduct(dVeta));
            Vector Vy_Vt = Vy.Field("_Vt"), Vy_S = Vy.Field("_S");
            Vector tmp2 = dVeta * Vy_Vt.GetTranspose();
            Vy_S = 1 / Vy_S;
            tmp2 = Vy_S.GetDiagTimesM(tmp2, GLOBAL::R) * Vy_Vt;
            Vector result2 = (tmp2 - Vy * (tmp2.GetTranspose() * tmp2)) * alpha2;
            
            *result = Uy * Dy * result2.GetTranspose() + Uy * xix.Field("dD") * Vy.GetTranspose() + result1 * Dy * Vy.GetTranspose();
            result->AddToFields("dU", result1);
            result->AddToFields("dD", xix.Field("dD"));
            result->AddToFields("dV", result2);
            
            if (IsEtaXiSameDir && HasHHR)
            {
                realdp nxix = std::sqrt(Metric(x, xix, xix));
                Vector beta(3);
                realdp *betaptr = beta.ObtainWriteEntireData();
                realdp EtatoXi = std::sqrt(Metric(x, etax, etax)) / nxix;
                betaptr[0] = std::sqrt(Metric(x, etax, etax) / Metric(y, *result, *result)) / EtatoXi;
                betaptr[1] = Metric(x, etax, etax);
                betaptr[2] = Metric(y, *result, *result) * EtatoXi * EtatoXi;
                etax.AddToFields("beta", beta);
                
                if (HasHHR)
                {
                    Vector betaresult(*result); ScalarTimesVector(y, betaptr[0] * EtatoXi, *result, &betaresult);
                    etax.AddToFields("betaTReta", betaresult);
                }
            }
            
            return *result;
        }
        
        Vector Ux = x.Field("U"), dUeta = etax.Field("dU"), Uy = y.Field("U"), dUxix = xix.Field("dU");
        Vector Dx = x.Field("D"), Dy = y.Field("D"), dDxix = xix.Field("dD");
        Vector Vx = x.Field("V"), dVeta = etax.Field("dV"), Vy = y.Field("V"), dVxix = xix.Field("dV");
        
        Vector Uy_Vt = Uy.Field("_Vt"), Uy_S = Uy.Field("_S");
        Vector tmp1(Uy_Vt); Uy_S.DiagTimesM(tmp1); tmp1 = Uy_Vt.GetTranspose() * tmp1;
        Vector tmp2 = dUxix / tmp1;
        Vector tmp3 = Uy.GetTranspose() * dUxix - dUxix.GetTranspose() * Uy;
        Vector Omega1 = tmp3.SYL(tmp1, tmp1);
        Vector result1 = tmp2 - Uy * (Uy.GetTranspose() * tmp2); /*dU, Uy * Omega1 + */

        Vector Vy_Vt = Vy.Field("_Vt"), Vy_S = Vy.Field("_S");
        Vector tmp4(Vy_Vt); Vy_S.DiagTimesM(tmp4); tmp4 = Vy_Vt.GetTranspose() * tmp4;
        Vector tmp5 = dVxix / tmp4;
        Vector tmp6 = Vy.GetTranspose() * dVxix - dVxix.GetTranspose() * Vy;
        Vector Omega2 = tmp6.SYL(tmp4, tmp4);
        Vector result2 = tmp5 - Vy * (Vy.GetTranspose() * tmp5); /*dV, Vy * Omega2 + */
        
        *result = Uy * Dy * result2.GetTranspose() + Uy * (dDxix + Omega1 * Dy - Dy * Omega2) * Vy.GetTranspose() + result1 * Dy * Vy.GetTranspose();
//        *result = Uy * Dy * result2.GetTranspose() + Uy * (dDxix) * Vy.GetTranspose() + result1 * Dy * Vy.GetTranspose();
        result->AddToFields("dU", result1);
        result->AddToFields("dD", dDxix + Omega1 * Dy - Dy * Omega2); /* + Omega1 * Dy - Dy * Omega2 */
        result->AddToFields("dV", result2);

        return *result;
    };

	Vector &FixedRankE::VecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
	{
        printf("Warning: FixedRankE::VecTranDiffRetAdjoint has not been overridden!\n");
        return Manifold::VecTranDiffRetAdjoint(x, etax, y, xiy, result);
	};

    Vector &FixedRankE::InverseVecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
    {
        Vector Ux = x.Field("U"), dUeta = etax.Field("dU"), Uy = y.Field("U"), dUxix = xix.Field("dU");
        Vector Dx = x.Field("D"), Dy = y.Field("D"), dDxix = xix.Field("dD");
        Vector Vx = x.Field("V"), dVeta = etax.Field("dV"), Vy = y.Field("V"), dVxix = xix.Field("dV");

//        Dy.SVDDecom();Dy.Field("_S").Print("Dy_S:");//----
        
        Vector Uy_Vt = Uy.Field("_Vt"), Uy_S = Uy.Field("_S");
        Vector tmp1(Uy_Vt); Uy_S.DiagTimesM(tmp1); tmp1 = Uy_Vt.GetTranspose() * tmp1;
        Vector tmp2 = dDxix * Dy.GetTranspose();
        Vector T1 = tmp2.SYL(tmp1, tmp1);
        Vector tmp3 = dUxix * Dx * Dx.GetTranspose();
        Vector tmp4 = Uy.GetTranspose() * tmp3;
        Vector tmp5 = (tmp3 - Ux * ((Uy.GetTranspose() * Ux ) % (tmp4 - T1 + T1.GetTranspose()))) * tmp1 / (Dy * Dy.GetTranspose());
        Vector dUresult = tmp5 - Uy * (Uy.GetTranspose() * tmp5);
        
        Vector dDresult = dDxix;
        
        Vector Vy_Vt = Vy.Field("_Vt"), Vy_S = Vy.Field("_S");
        Vector tmp6(Vy_Vt); Vy_S.DiagTimesM(tmp6); tmp6 = Vy_Vt.GetTranspose() * tmp6;
        Vector tmp7 = Dy.GetTranspose() * dDxix;
        Vector T2 = tmp7.SYL(tmp6, tmp6);
        Vector tmp8 = dVxix * Dx.GetTranspose() * Dx;
        Vector tmp9 = Vy.GetTranspose() * tmp8;
        Vector tmp10 = (tmp8 - Vx * ((Vy.GetTranspose() * Vx) % ( tmp9 - T2.GetTranspose() + T2))) * tmp6 / (Dy.GetTranspose() * Dy);
        Vector dVresult = tmp10 - Vy * (Vy.GetTranspose() * tmp10);
        *result = Uy * Dy * dVresult.GetTranspose() + Uy * dDresult * Vy.GetTranspose() + dUresult * Dy * Vy.GetTranspose();
        result->AddToFields("dU", dUresult);
        result->AddToFields("dD", dDresult);
        result->AddToFields("dV", dVresult);
        
        return *result;
    };

    Vector &FixedRankE::InverseVecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result, bool IsEtaXiSameDir) const
    {
        Vector Ux = x.Field("U"), dUeta = etax.Field("dU"), Uy = y.Field("U"), dUxiy = xiy.Field("dU");
        Vector Dx = x.Field("D"), dDeta = etax.Field("dD"), Dy = y.Field("D"), dDxiy = xiy.Field("dD");
        Vector Vx = x.Field("V"), dVeta = etax.Field("dV"), Vy = y.Field("V"), dVxiy = xiy.Field("dV");
//        Dy.SVDDecom(); Dy.Field("_S").Print("Dy_S:");//---
//        Dx.SVDDecom(); Dx.Field("_S").Print("Dx_S:");//---
        
        Vector Uy_Vt = Uy.Field("_Vt"), Uy_S = Uy.Field("_S");
        Vector tmp1(Uy_Vt); Uy_S.DiagTimesM(tmp1); tmp1 = Uy_Vt.GetTranspose() * tmp1;
        Vector tmp2 = dUxiy * tmp1;
        Vector tmp3 = Uy.GetTranspose() * tmp2;
        Vector P1 = tmp2 - Uy * tmp3;
        Vector A1 = (-1) * (Ux.GetTranspose() * Uy) % (Ux.GetTranspose() * P1);
        
//        std::cout << "dUxiynorm:" << dUxiy.Fnorm() << std::endl;//---
//        std::cout << "tmp1:" << tmp1.Fnorm() << std::endl;//---
//        std::cout << "tmp2:" << tmp2.Fnorm() << std::endl;//---
//        std::cout << "tmp3:" << tmp3.Fnorm() << std::endl;//---
//
//        std::cout << "A1norm:" << A1.Fnorm() << std::endl;//---
//        std::cout << "UxPnorm:" << (Ux.GetTranspose() * P1).Fnorm() << std::endl;//---
//        std::cout << "UxUynorm:" << (Ux.GetTranspose() * Uy).Fnorm() << std::endl;//---
        Vector result1 = Uy * A1 + P1;
        Vector tmp4 = A1 - A1.GetTranspose();
        Vector Omega1 = tmp4.SYL(tmp1, tmp1);
        
        
        Vector Vy_Vt = Vy.Field("_Vt"), Vy_S = Vy.Field("_S");
        Vector tmp5(Vy_Vt); Vy_S.DiagTimesM(tmp5); tmp5 =Vy_Vt.GetTranspose() * tmp5;
        Vector tmp6 = dVxiy * tmp5;
        Vector tmp7 = Vy.GetTranspose() * tmp6;
        Vector P2 = tmp6 - Vy * tmp7;
        Vector A2 = (-1) * (Vx.GetTranspose() * Vy) % (Vx.GetTranspose() * P2);
        Vector result2 = Vy * A2 + P2;
        Vector tmp8 = A2 - A2.GetTranspose();
        Vector Omega2 = tmp8.SYL(tmp5, tmp5);
//        std::cout << "here:" << (dDxiy - Omega1 * Dy + Dy * Omega2).Fnorm() << ":" << result1.Fnorm() << ":" << result2.Fnorm() << std::endl;//---
        
        *result = Ux * Dx * result2.GetTranspose() + Ux * (dDxiy - Omega1 * Dy + Dy * Omega2) * Vx.GetTranspose() + result1 * Dx * Vx.GetTranspose();
        result->AddToFields("dU", result1);
        result->AddToFields("dD", dDxiy - Omega1 * Dy + Dy * Omega2);
        result->AddToFields("dV", result2);
        
        return *result;
    };

	Vector &FixedRankE::Projection(const Variable &x, const Vector &etax, Vector *result) const
	{
        return ExtrProjection(x, etax, result);
	};

    Vector &FixedRankE::ExtrProjection(const Variable &x, const Vector &etax, Vector *result) const
    {
        Vector Ux = x.Field("U"), Vx = x.Field("V"), Dx = x.Field("D");
        Vector MV = etax * Vx;
        Vector dD = Ux.GetTranspose() * MV;
        Vector dU = (MV - Ux * dD) / Dx;
        Vector dV = (etax.GetTranspose() * Ux - Vx * dD.GetTranspose()) / Dx.GetTranspose();
        
//        dU.RandGaussian();//---
//        dU = dU - Ux * (Ux.GetTranspose() * dU);//---
//        dU.Print("h1:");//---
//        dD.Print("h2:");//---
//        dV.Print("h3:");//--
        
        *result = Ux * Dx * dV.GetTranspose() + Ux * dD * Vx.GetTranspose() + dU * Dx * Vx.GetTranspose();
        result->AddToFields("dU", dU);
        result->AddToFields("dD", dD);
        result->AddToFields("dV", dV);
        return *result;
    };

	Vector &FixedRankE::EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result)  const
	{
        Projection(x, egf, result);
        if(prob->GetUseHess())
        {
            x.AddToFields("EGrad", egf);
        }
        return *result;
	};

	Vector &FixedRankE::EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const
	{
        Vector xix(EMPTYEXTR);
        Projection(x, exix, &xix);
        Vector segf = x.Field("EGrad");
        Vector dU = etax.Field("dU"), dV = etax.Field("dV");
        Vector MdV = segf * dV, MTdU = segf.GetTranspose() * dU;
        Vector U = x.Field("U"), D = x.Field("D"), V = x.Field("V");
  
        Vector dUxix = xix.Field("dU"), dVxix = xix.Field("dV"), dDxix = xix.Field("dD");
        dUxix = dUxix + (MdV - U * (U.GetTranspose() * MdV)) / D;
        
        dVxix = dVxix + (MTdU - V * (V.GetTranspose() * MTdU)) / D.GetTranspose();
        
        *result = U * D * dVxix.GetTranspose() + U * dDxix * V.GetTranspose() + dUxix * D * V.GetTranspose();
        result->AddToFields("dU", dUxix);
        result->AddToFields("dD", dDxix);
        result->AddToFields("dV", dVxix);
        return *result;
	};
}; /*end of ROPTLIB namespace*/
