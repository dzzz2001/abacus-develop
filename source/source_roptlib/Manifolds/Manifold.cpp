
#include "Manifolds/Manifold.h"

/*Define the namespace*/
namespace ROPTLIB{

	Manifold::~Manifold(void)
	{
	};

	realdp Manifold::Metric(const Variable &x, const Vector &etax, const Vector &xix) const
	{
    #ifdef CHECKMANIFOLDOVERRIDDEN
            printf("Manifold::Metric has not been overridden!\n");
    #endif
        return etax.DotProduct(xix);
	};

	Vector &Manifold::LinearOPEEta(const Variable &x, const LinearOPE &Hx, const Vector &etax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::LinearOPEEta has not been overridden!\n");
        #endif

        /*even if etax is a complex element, the real reparameterizations are used. In other words, we viewed it is
        2 n dimension real data. We have to convert it to real element to apply the linear opeartor. */
        bool etaxiscomplex = etax.Getiscomplex();
        etax.Setiscomplex(false);
        Vector etaxreshape = etax; etaxreshape.Reshape(etax.Getlength());
        result->AlphaABaddBetaThis(1, Hx, GLOBAL::N, etaxreshape, GLOBAL::N, 0);
        etax.Setiscomplex(etaxiscomplex);
        result->Setiscomplex(etaxiscomplex);
        result->Reshape(etax.Getrow(), etax.Getcol(), etax.Getnum());
        return *result;
	};

    Vector &Manifold::Projection(const Variable &x, const Vector &etax, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::Projection has not been overridden!\n");
        #endif
        if (!IsIntrApproach)
        {
            return this->ExtrProjection(x, etax, result);
        }
        else
        {
            return this->IntrProjection(x, etax, result);
        }
    };

    Vector &Manifold::IntrProjection(const Variable &x, const Vector &etax, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::IntrProjection has not been overridden!\n");
        #endif
        *result = etax;
        return *result;
    };

    Vector &Manifold::ExtrProjection(const Variable &x, const Vector &etax, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ExtrProjection has not been overridden!\n");
        #endif
        *result = etax;
        return *result;
    };

	Vector &Manifold::ScalarTimesVector(const Variable &x, const realdp &scalar, const Vector &etax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ScalarTimesVector has not been overridden!\n");
        #endif
        *result = etax;
        result->ScalarTimesThis(scalar);
        return *result;
	};

    Vector &Manifold::ScalarVectorAddVector(const Variable &x, const realdp &scalar, const Vector &etax, const Vector &xix, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ScalarVectorAddVector has not been overridden!\n");
        #endif
        *result = xix;
        result->AlphaXaddThis(scalar, etax);
        return *result;
    };

    Vector &Manifold::VectorLinearCombination(const Variable &x, realdp scalar1, const Vector &etax, realdp scalar2, const Vector &xix, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::VectorLinearCombination has not been overridden!\n");
        #endif
        *result = xix;
        result->ScalarTimesThis(scalar2);
        result->AlphaXaddThis(scalar1, etax);
        
        return *result;
    };

    Variable &Manifold::Retraction(const Variable &x, const Vector &etax, Variable *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::Retraction has not been overridden!\n");
        #endif
        *result = x; result->AlphaXaddThis(1, etax); /*result = x + etax*/
        return *result;
    };

    Vector &Manifold::InvRetraction(const Variable &x, const Variable &y, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::InvRetraction has not been overridden!\n");
        #endif
        *result = y; result->AlphaXaddThis(-1, x); /*result = y - x*/
        
        return ExtrProjection(x, *result, result);
    };

	Vector &Manifold::VecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::VecTranDiffRetAdjoint has not been overridden!\n");
        #endif
        
        return Projection(x, xiy, result);
	};

	Vector &Manifold::VecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result, bool IsEtaXiSameDir) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::VecTranDiffRet has not been overridden!\n");
        #endif
        
        Projection(y, xix, result);

        if (IsEtaXiSameDir && HasHHR)
        {
            Vector beta(3);
            realdp *betaptr = beta.ObtainWriteEntireData();
            realdp EtatoXi = sqrt(Metric(x, etax, etax) / Metric(x, xix, xix));
            betaptr[0] = std::sqrt(Metric(x, etax, etax) / Metric(x, *result, *result)) / EtatoXi;
            betaptr[1] = Metric(x, etax, etax);
            betaptr[2] = Metric(x, *result, *result) * EtatoXi * EtatoXi;
            etax.AddToFields("beta", beta);
            
            if (HasHHR)
            {
                etax.AddToFields("betaTReta", (*result) * (betaptr[0] * EtatoXi));
            }
        }
        
        return *result;
	};

    Vector &Manifold::InverseVecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::InverseVecTranDiffRetAdjoint has not been overridden!\n");
        #endif
        
        return Projection(y, xix, result);
    };

    Vector &Manifold::InverseVecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result, bool IsEtaXiSameDir) const
    {
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::InverseVecTranDiffRet has not been overridden!\n");
        #endif
        
        return Projection(x, xiy, result);
    };


	realdp Manifold::Beta(const Variable &x, const Vector &etax) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::Beta has not been overridden!\n");
        #endif
		return 1;
	};

	realdp Manifold::Dist(const Variable &x1, const Variable &x2) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::Dist has not been overridden!\n");
        #endif
		return (x1 - x2).Fnorm();
	};

	Vector &Manifold::VectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::VectorTransport has not been overridden!\n");
        #endif
        if (HasHHR)
            return LCVectorTransport(x, etax, y, xix, result);
        
        return Projection(y, xix, result);
	};

	Vector &Manifold::InverseVectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::InverseVectorTransport has not been overridden!\n");
        #endif
		if (HasHHR)
			return LCInverseVectorTransport(x, etax, y, xiy, result);
		
		return Projection(x, xiy, result);
	};

	LinearOPE &Manifold::HInvTran(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, integer start, integer end, LinearOPE *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::HInvTran has not been overridden!\n");
        #endif
		if (HasHHR)
			return LCHInvTran(x, etax, y, Hx, start, end, result);
		
        *result = Hx;
        return *result;
	};

	LinearOPE &Manifold::TranH(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, integer start, integer end, LinearOPE *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::TranH has not been overridden!\n");
        #endif
		if (HasHHR)
			return LCTranH(x, etax, y, Hx, start, end, result);
		
        *result = Hx;
		return *result;
	};

	LinearOPE &Manifold::TranHInvTran(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, LinearOPE *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::TranHInvTran has not been overridden!\n");
        #endif
        
		if (HasHHR)
			return LCTranHInvTran(x, etax, y, Hx, result);
        
        *result = Hx;
        return *result;
	};

	LinearOPE &Manifold::HaddScaledRank1OPE(const Variable &x, const LinearOPE &Hx, realdp scalar, const Vector &etax, const Vector &xix, LinearOPE *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::HaddScaledRank1OPE has not been overridden!\n");
        #endif
        /*even if etax and xix are complex elements, the real reparameterizations are used. In other words, we viewed it is
        2 n dimension real data. Therefore, the low rank update uses real operations. */
        bool etaxiscomplex = etax.Getiscomplex(), xixiscomplex = xix.Getiscomplex();
        etax.Setiscomplex(false);
        xix.Setiscomplex(false);
        *result = Hx;
        Vector etaxreshape = etax; etaxreshape.Reshape(etax.Getlength());
        Vector xixreshape = xix; xixreshape.Reshape(xix.Getlength());
        result->HaddRankone(scalar, etaxreshape, xixreshape);
        etax.Setiscomplex(etaxiscomplex);
        xix.Setiscomplex(xixiscomplex);
        return *result;
	};

	Vector &Manifold::ObtainEtaxFlat(const Variable &x, const Vector &etax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ObtainEtaxFlat has not been overridden!\n");
        #endif
        *result = etax;
        return *result;
	};

	Vector &Manifold::ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ObtainIntr has not been overridden!\n");
        #endif
        *result = etax;
        return *result;
	};

	Vector &Manifold::ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ObtainExtr has not been overridden!\n");
        #endif
        *result = intretax;
        return *result;
	};

	Vector &Manifold::ObtainNorVerIntr(const Variable &x, const Vector &etax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ObtainNorVerIntr has not been overridden!\n");
        #endif
        *result = etax;
        return *result;
	};

	Vector &Manifold::ObtainNorVerExtr(const Variable &x, const Vector &intretax, Vector *result) const
	{
        #ifdef CHECKMANIFOLDOVERRIDDEN
                printf("Manifold::ObtainNorVerExtr has not been overridden!\n");
        #endif
        *result = intretax;
        return *result;
	};

    void Manifold::CheckParams(void) const
    {
        printf("GENERAL PARAMETERS:\n");
        printf("name          :%15s,\n", name.c_str());
        printf("IsIntrApproach:%15d,\t", IsIntrApproach);
        printf("IntrinsicDim  :%15d,\n", IntrinsicDim);
        printf("ExtrinsicDim  :%15d,\t", ExtrinsicDim);
        printf("HasHHR        :%15d,\n", HasHHR);
    };

    void Manifold::SetParams(PARAMSMAP params)
    {
        PARAMSMAP::iterator iter;
        for (iter = params.begin(); iter != params.end(); iter++)
        {
            if (iter->first == static_cast<std::string> ("HasHHR"))
            {
                SetHasHHR(((static_cast<integer> (iter->second)) != 0));
            }
        }
    };

	void Manifold::CheckIntrExtr(Variable x) const
	{
		printf("==============Check Intrinsic/Extrinsic transform=========\n");
		Vector exetax(EMPTYEXTR);
		Vector inetax(EMPTYINTR);
		x.Print("x");
		exetax.RandGaussian();
		ExtrProjection(x, exetax, &exetax);
		exetax.Print("exetax1");
		 ObtainIntr(x, exetax, &inetax);
        bool Isintr = GetIsIntrinsic();
        SetIsIntrApproach(false);
		printf("extr inp:%g\n", Metric(x, exetax, exetax));
        SetIsIntrApproach(true);
		printf("intr inp:%g\n", Metric(x, inetax, inetax));
        SetIsIntrApproach(Isintr);
		inetax.Print("inetax1");
		ObtainExtr(x, inetax, &exetax);
		exetax.Print("exetax2");
		ObtainIntr(x, exetax, &inetax);
		inetax.Print("inetax2");
		printf("exeta1 and inetax1 should approximately equal exetax2 and inetax2 respectively!\n");
	};

	void Manifold::CheckRetraction(Variable x) const
	{
		printf("==============Check Retraction=========\n");
		Vector etax(EMPTYEXTR), FDetax(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);
        ScalarTimesVector(x, 1.0 / sqrt(Metric(x, etax, etax)), etax, &etax);
		x.Print("x:");
		etax.Print("etax:");
#ifdef SINGLE_PRECISION
		realdp eps = static_cast<realdp> (1e-3);
#else
        realdp eps = static_cast<realdp> (1e-5);
#endif
		Variable y(x);
		ScalarTimesVector(x, eps, etax, &etax);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			Retraction(x, inetax, &y);
		}
		else
		{
			Retraction(x, etax, &y);
		}
        FDetax = (y - x) / eps;
		FDetax.Print("FDetax:");

		printf("etax should approximately equal FDetax = (R(eps etax)-R(etax))/eps!\n");
	};

	void Manifold::CheckVecTranDiffRet(Variable x, bool IsEtaXiSameDir) const
	{
		printf("==============Check Differentiated Retraction=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetax(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);
		if (IsEtaXiSameDir)
		{
            xix = etax;
		}
		else
		{
			xix.RandGaussian();
			ExtrProjection(x, xix, &xix);
		}
		x.Print("x:");
		etax.Print("etax:");
//        (x.GetTranspose() * etax).Print("xt * etax:");//---
//        (x.GetTranspose() * xix).Print("xt * xix:");//---
		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetax(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			ObtainIntr(x, xix, &inxix);
            std::cout << "************************************************************************************************************" << std::endl;
			Retraction(x, inetax, &y);
			VecTranDiffRet(x, inetax, y, inxix, &inzetax, IsEtaXiSameDir);
			ObtainExtr(y, inzetax, &zetax);
		}
		else
		{
			Retraction(x, etax, &y);
			VecTranDiffRet(x, etax, y, xix, &zetax, IsEtaXiSameDir);
		}
		y.Print("y:");
		zetax.Print("zetax:");
//        (y.GetTranspose() * zetax).Print("yt * zeta:");//---
		Variable yeps(x);
		realdp eps = static_cast<realdp> (1e-5);
        ScalarVectorAddVector(x, eps, xix, etax, &etax);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			Retraction(x, inetax, &yeps);
		}
		else
		{
			Retraction(x, etax, &yeps);
		}
        zetax = (yeps - y) / eps;
		ExtrProjection(y, zetax, &zetax);
		zetax.Print("FDzetax:");
//        (y.GetTranspose() * zetax).Print("yt * FDzeta:");//---
		printf("zetax = T_{R_etax} xix should approximately equal FDzetax = (R(etax+eps xix) - R(etax))/eps!\n");
	};

    void Manifold::CheckInverseVecTranDiffRet(Variable x) const
    {
        printf("==============Check Inverse Differentiated Retraction=========\n");
        Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetax(EMPTYEXTR);
        etax.RandGaussian();
        ExtrProjection(x, etax, &etax);
        xix.RandGaussian();
        ExtrProjection(x, xix, &xix);
        x.Print("x:");
        etax.Print("etax:");
        Variable y(x);
        if (IsIntrApproach)
        {
            Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetax(EMPTYINTR);
            ObtainIntr(x, etax, &inetax);
            ObtainIntr(x, xix, &inxix);
            xix.Print("xix:");
            std::cout << "************************************************************************************************************" << std::endl;
            Retraction(x, inetax, &y);
            VecTranDiffRet(x, inetax, y, inxix, &inzetax);
            ObtainExtr(y, inzetax, &zetax);
            zetax.Print("T_R xix:");
            InverseVecTranDiffRet(x, inetax, y, inzetax, &inzetax);
            ObtainExtr(x, inzetax, &zetax);
            zetax.Print("T_R^{-1} T_R xix:");
        }
        else
        {
            Retraction(x, etax, &y);
            y.Print("y:");
            VecTranDiffRet(x, etax, y, xix, &zetax);
            xix.Print("xix:");
//            (x.GetTranspose() * xix).Print("xt * xix");//---
            zetax.Print("T_R xix:");
//            (y.GetTranspose() * zetax).Print("yt * zetax:");//---
            InverseVecTranDiffRet(x, etax, y, zetax, &zetax);
//            ExtrProjection(x, zetax, &zetax);
            zetax.Print("T_R^{-1} T_R xix:");
//            ExtrProjection(x, zetax, &zetax);
//            zetax.Print("T_R^{-1} T_R xix 2:", false);
            VecTranDiffRet(x, etax, y, zetax, &zetax);
            zetax.Print("T_R T_R^{-1} T_R xix:");
//            (x.GetTranspose() * zetax).Print("xt * T_R^{-1} T_R xix:");//---
        }
    };

    void Manifold::CheckInverseVecTranDiffRetAdjoint(Variable x) const
    {
        printf("==============Check the Adjoint Operator of the Inverse Differentiated Retraction=========\n");
        Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetay(EMPTYEXTR), etay(EMPTYEXTR);
        etax.RandGaussian();
        ExtrProjection(x, etax, &etax);
        xix.RandGaussian();
        ExtrProjection(x, xix, &xix);
//        x.Print("x:");
//        etax.Print("etax:");
        Variable y(x);
        if (IsIntrApproach)
        {
            Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetay(EMPTYINTR), inetay(EMPTYINTR);
            ObtainIntr(x, etax, &inetax);
            ObtainIntr(x, xix, &inxix);
//            xix.Print("xix:");
            Retraction(x, inetax, &y);
            etay.RandGaussian();
            ExtrProjection(y, etay, &inetay);
            
            InverseVecTranDiffRetAdjoint(x, inetax, y, inxix, &inzetay);
            printf("<T_R^{-sharp} xix, etay>: %.10e\n", Metric(y, inzetay, inetay));
            InverseVecTranDiffRet(x, inetax, y, inetay, &inetay);
            printf("<xix, T_R^{-1} etay>: %.10e\n", Metric(x, inxix, inetay));
        }
        else
        {
            Retraction(x, etax, &y);
            etay.RandGaussian();
            ExtrProjection(y, etay, &etay);
            
            InverseVecTranDiffRetAdjoint(x, etax, y, xix, &zetay);
            printf("<T_R^{-sharp} xix, etay>: %.10e\n", Metric(y, zetay, etay));
            InverseVecTranDiffRet(x, etax, y, etay, &etay);
            printf("<xix, T_R^{-1} etay>: %.10e\n", Metric(x, xix, etay));
        }
    };

	void Manifold::CheckLockingCondition(Variable x) const
	{
		printf("==============Check Locking Condition=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetax(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);
		ScalarTimesVector(x, 1, etax, &xix); /* genrandreal() + static_cast<realdp> (0.5) */
		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetax(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			ObtainIntr(x, xix, &inxix);
			Retraction(x, inetax, &y);
			VecTranDiffRet(x, inetax, y, inxix, &inzetax, true);
			if (inetax.FieldsExist("beta"))
			{
                Vector beta = inetax.Field("beta");
                const realdp *betav = beta.ObtainReadData();
                printf("beta = |etax| / |T_{etax} etax|: %g\n", betav[0]);
			}
			else
			{
				printf("beta: %d\n", 1);
			}
			printf("|xix| / |T_{etax} xix|:%g\n", sqrt(Metric(x, inxix, inxix) / Metric(x, inzetax, inzetax)));
			ScalarTimesVector(x, sqrt(Metric(x, inxix, inxix) / Metric(x, inzetax, inzetax)), inzetax, &inzetax);
			ObtainExtr(y, inzetax, &zetax);
			zetax.Print("Beta VecTranDiffRet zetax:");
			VectorTransport(x, inetax, y, inxix, &inzetax);
            
			ObtainExtr(y, inzetax, &zetax);
			zetax.Print("Vector Transport zetax:");
		}
		else
		{
			Retraction(x, etax, &y);
			VecTranDiffRet(x, etax, y, xix, &zetax, true);
			if (etax.FieldsExist("beta"))
			{
                Element beta = etax.Field("beta");
                const realdp *betav = beta.ObtainReadData();
                printf("beta = |etax| / |T_{etax} etax|: %g\n", betav[0]);
			}
			else
			{
				printf("beta: %d\n", 1);
			}
			printf("|xix| / |T_{etax} xix|:%g\n", sqrt(Metric(x, xix, xix) / Metric(y, zetax, zetax)));
			ScalarTimesVector(y, sqrt(Metric(x, xix, xix) / Metric(y, zetax, zetax)), zetax, &zetax);
			zetax.Print("Beta VecTranDiffRet zetax:");
			VectorTransport(x, etax, y, xix, &zetax);
			zetax.Print("Vector Transport zetax:");
		}
		printf("Beta VecTranDiffRet zetax should approximately equal Vector Transport zetax!\n");
	};

	void Manifold::CheckVecTranDiffRetAdjoint(Variable x) const
	{
		printf("==============Check CoTangentVector=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetay(EMPTYEXTR), xiy(EMPTYEXTR), zetax(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);

		xix.RandGaussian();
		ExtrProjection(x, xix, &xix);

		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetay(EMPTYINTR), inxiy(EMPTYINTR), inzetax(EMPTYINTR);
            ObtainIntr(x, etax, &inetax);
			ObtainIntr(x, xix, &inxix);
			Retraction(x, inetax, &y);
			VecTranDiffRet(x, inetax, y, inxix, &inzetay, false);
			ObtainExtr(y, inzetay, &zetay);

			xiy.RandGaussian();
            ExtrProjection(y, xiy, &xiy);
			ObtainIntr(y, xiy, &inxiy);
			printf("<xiy, T_{R_{eta}} xix>:%g\n", Metric(y, inxiy, inzetay));

			VecTranDiffRetAdjoint(x, inetax, y, inxiy, &inzetax);
			ObtainExtr(x, inzetax, &zetax);
			printf("C(x, etax, xiy) [xix]:%g\n", Metric(x, inzetax, inxix));
		}
		else
		{
			Retraction(x, etax, &y);
			VecTranDiffRet(x, etax, y, xix, &zetay, false);
			xiy.RandGaussian();
			ExtrProjection(y, xiy, &xiy);
			ScalarTimesVector(y, sqrt(Metric(y, xiy, xiy)), xiy, &xiy);
			printf("<xiy, T_{R_{eta}} xix>:%g\n", Metric(y, xiy, zetay));
			VecTranDiffRetAdjoint(x, etax, y, xiy, &zetax);
			printf("C(x, etax, xiy) [xix]:%g\n", Metric(x, zetax, xix));
		}
		printf("<xiy, T_{R_{eta}} xix> should approximately equal C(x, etax, xiy) [xix]!\n");
	};

	void Manifold::CheckIsometryofVectorTransport(Variable x) const
	{
		printf("==============Check Isometry of the Vector Transport=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetay(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);

		xix.RandGaussian();
		ExtrProjection(x, xix, &xix);

		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetay(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			ObtainIntr(x, xix, &inxix);
			Retraction(x, inetax, &y);
			VectorTransport(x, inetax, y, inxix, &inzetay);
			printf("Before vector transport:%g, After vector transport:%g\n", Metric(x, inxix, inxix), Metric(y, inzetay, inzetay));
		}
		else
		{
			Retraction(x, etax, &y);
			VectorTransport(x, etax, y, xix, &zetay);
			y.Print("y:");
			zetay.Print("zetay:");
			printf("Before vector transport:%g, After vector transport:%g\n", Metric(x, xix, xix), Metric(y, zetay, zetay));
		}
		printf("|xix| (Before vector transport) should approximately equal |T_{R_etax} xix| (After vector transport)\n");
	};

	void Manifold::CheckIsometryofInvVectorTransport(Variable x) const
	{
		printf("==============Check Isometry of the Inverse Vector Transport=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetay(EMPTYEXTR);

		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);

		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetay(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			Retraction(x, inetax, &y);
			zetay.RandGaussian();
			ExtrProjection(y, zetay, &zetay);
			ScalarTimesVector(y, sqrt(Metric(y, zetay, zetay)), zetay, &zetay);
			ObtainIntr(y, zetay, &inzetay);

			InverseVectorTransport(x, inetax, y, inzetay, &inxix);
			printf("Before inverse vector transport:%g, After inverse vector transport:%g\n", Metric(y, inzetay, inzetay), Metric(x, inxix, inxix));
		}
		else
		{
			Retraction(x, etax, &y);
			zetay.RandGaussian();
			ExtrProjection(y, zetay, &zetay);
			InverseVectorTransport(x, etax, y, zetay, &xix);
			x.Print("x:");
			xix.Print("xix:");
			printf("Before inverse vector transport:%g, After inverse vector transport:%g\n", Metric(y, zetay, zetay), Metric(x, xix, xix));
		}
		printf("|zetay| (Before inverse vector transport) should approximately equal |T_{R_etax}^{-1} zetay| (After inverse vector transport)\n");
	};

	void Manifold::CheckVecTranComposeInverseVecTran(Variable x) const
	{
		printf("==============Check Vector Transport Compose Inverse Vector Transport=========\n");
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR), zetay(EMPTYEXTR);

		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);
		xix.RandGaussian();
		ExtrProjection(x, xix, &xix);

		Variable y(x);
		if (IsIntrApproach)
		{
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR), inzetay(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			Retraction(x, inetax, &y);
			ObtainIntr(x, xix, &inxix);
			xix.Print("xix:");
			VectorTransport(x, inetax, y, inxix, &inzetay);
			InverseVectorTransport(x, inetax, y, inzetay, &inxix);
			ObtainExtr(x, inxix, &xix);
			xix.Print("T^{-1} ciric T xix:");
			printf("xix and T^{-1} ciric T xix should be similar!\n");
		}
		else
		{
			Retraction(x, etax, &y);
			xix.Print("xix:");
			VectorTransport(x, etax, y, xix, &zetay);
			InverseVectorTransport(x, etax, y, zetay, &xix);
			xix.Print("T^{-1} ciric T xix:");
			printf("xix and T^{-1} ciric T xix should be similar!\n");
		}
	};

	void Manifold::CheckTranHInvTran(Variable x) const
	{
		printf("==============Check Transport of a Hessian approximation=========\n");
		Vector etax(EMPTYEXTR);
		Variable y(x);

		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);

		if (IsIntrApproach)
		{
            LinearOPE Hx(EMPTYINTR.Getlength(), EMPTYINTR.Getlength()), result(EMPTYINTR.Getlength(), EMPTYINTR.Getlength());
			Vector inetax(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			Retraction(x, inetax, &y);
			Hx.ScaledIdOPE();
			Hx.Print("Hx before:");
			TranHInvTran(x, inetax, y, Hx, &result);
			result.Print("Hx after:");
		}
		else
		{
            LinearOPE Hx(EMPTYEXTR.Getlength(), EMPTYEXTR.Getlength()), result(EMPTYEXTR.Getlength(), EMPTYEXTR.Getlength());
			Hx.ScaledIdOPE();
			Hx.Print("Hx before:");
			Retraction(x, etax, &y);
			Vector zetay1(EMPTYEXTR), zetay2(EMPTYEXTR);
			zetay1.RandGaussian();
			ExtrProjection(y, zetay1, &zetay1);
			TranHInvTran(x, etax, y, Hx, &result);
			result.Print("Hx after:");
			zetay1.Print("zetay:");
            LinearOPEEta(y, result, zetay1, &zetay2);
			zetay2.Print("Hx zetay:");
		}
	};

	void Manifold::CheckHaddScaledRank1OPE(Variable x) const
	{
		printf("==============Check Rank one Update to a Hessian Approximation=========\n");
		realdp scalar = 1.0;
		Vector etax(EMPTYEXTR), xix(EMPTYEXTR);
		etax.RandGaussian();
		ExtrProjection(x, etax, &etax);
		xix.RandGaussian();
		ExtrProjection(x, xix, &xix);

		if (IsIntrApproach)
		{
            LinearOPE Hx(EMPTYINTR.Getlength(), EMPTYINTR.Getlength()), result(EMPTYINTR.Getlength(), EMPTYINTR.Getlength());
			Vector inetax(EMPTYINTR), inxix(EMPTYINTR);
			ObtainIntr(x, etax, &inetax);
			ObtainIntr(x, xix, &inxix);
			Hx.ScaledIdOPE();
			Hx.Print("Hx before:");
			HaddScaledRank1OPE(x, Hx, scalar, inetax, inxix, &result);
			inetax.Print("etax:");
			inxix.Print("xix:");
			result.Print("Hx after:");
		}
		else
		{
            LinearOPE Hx(EMPTYEXTR.Getlength(), EMPTYEXTR.Getlength()), result(EMPTYEXTR.Getlength(), EMPTYEXTR.Getlength());
			Hx.ScaledIdOPE();
			Hx.Print("Hx before:");
			HaddScaledRank1OPE(x, Hx, scalar, etax, xix, &result);
			etax.Print("etax:");
			xix.Print("xix:");
			result.Print("Hx after:");
		}
	};

    void Manifold::Obtainnu1nu2forLC(const Variable &x, const Vector &etax, const Variable &y) const
    {
        Vector eps1(etax), nu1(etax), nu2(etax);
        
        if (!etax.FieldsExist("beta") || !etax.FieldsExist("betaTReta"))
        {
            VecTranDiffRet(x, etax, y, etax, &eps1, true);
        }
        Vector TRetaVector = etax.Field("betaTReta");
        HasHHR = false; VectorTransport(x, etax, y, etax, &eps1); HasHHR = true;
        ScalarTimesVector(y, sqrt(Metric(x, etax, etax) / Metric(y, eps1, eps1)), eps1, &eps1); /*note \|etax\|_x = \|TRetaVector\|_y*/
        Element tau1tau2(2);
        realdp *tau1tau2ptr = tau1tau2.ObtainWriteEntireData();
        ScalarTimesVector(y, 2.0, eps1, &nu1);
        VectorLinearCombination(y, -1.0, eps1, -1.0, TRetaVector, &nu2);
        tau1tau2ptr[0] = Metric(y, nu1, nu1);
        tau1tau2ptr[0] = (std::fabs(tau1tau2ptr[0]) < 10 * std::numeric_limits<realdp>::epsilon()) ? 0 : static_cast<realdp> (2) / tau1tau2ptr[0]; // static_cast<realdp> (2) / Metric(y, nu1, nu1);
        tau1tau2ptr[1] = Metric(y, nu2, nu2);
        tau1tau2ptr[1] = (std::fabs(tau1tau2ptr[1]) < 10 * std::numeric_limits<realdp>::epsilon()) ? 0 : static_cast<realdp> (2) / tau1tau2ptr[1]; // static_cast<realdp> (2) / Metric(y, nu2, nu2)
        
        etax.AddToFields("tau1tau2", tau1tau2);
        etax.AddToFields("nu1", nu1);
        etax.AddToFields("nu2", nu2);
    };

    Vector &Manifold::LCVectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
    {
        if (!etax.FieldsExist("tau1tau2"))
        {
            Obtainnu1nu2forLC(x, etax, y);
        }
        HasHHR = false; VectorTransport(x, etax, y, xix, result); HasHHR = true;
        ScalarTimesVector(y, sqrt(Metric(x, xix, xix) / Metric(y, *result, *result)), *result, result);
        
//        result->Print("result1:");//--
        Element tau1tau2 = etax.Field("tau1tau2");
        const realdp *tau1tau2ptr = tau1tau2.ObtainReadData();
        Vector nu1 = etax.Field("nu1");
        Vector nu2 = etax.Field("nu2");
        realdp temp = - Metric(y, *result, nu1);
        ScalarVectorAddVector(y, temp * tau1tau2ptr[0], nu1, *result, result);
//        result->Print("result2:");//--
        temp = -Metric(y, *result, nu2);
//        nu1.Print("nu1:");//---
//        nu2.Print("nu2:");//---
//        tau1tau2.Print("tau1tau2:");//---
//        std::cout << "temp:" << temp << ", :" << tau1tau2ptr[1] << std::endl;//--
        ScalarVectorAddVector(y, temp * tau1tau2ptr[1], nu2, *result, result);
//        xix.Print("xix:");//---
//        result->Print("result3:");//--
        return *result;
    };

    Vector &Manifold::LCInverseVectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
    {
        if (!etax.FieldsExist("tau1tau2"))
        {
            Obtainnu1nu2forLC(x, etax, y);
        }
        
        Element tau1tau2 = etax.Field("tau1tau2");
        const realdp *tau1tau2ptr = tau1tau2.ObtainReadData();
        Vector nu1 = etax.Field("nu1");
        Vector nu2 = etax.Field("nu2");
        realdp temp = -Metric(y, xiy, nu2);
        VectorLinearCombination(y, temp * tau1tau2ptr[1], nu2, 1, xiy, result);
        temp = -Metric(y, *result, nu1);
        ScalarVectorAddVector(y, temp * tau1tau2ptr[0], nu1, *result, result);
        
        HasHHR = false; InverseVectorTransport(x, etax, y, *result, result); HasHHR = true;
        return *result;
    };

    LinearOPE &Manifold::LCHInvTran(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, integer start, integer end, LinearOPE *result) const
    {
        if (!etax.FieldsExist("tau1tau2"))
        {
            Obtainnu1nu2forLC(x, etax, y);
        }
        Element tau1tau2 = etax.Field("tau1tau2");
        const realdp *tau1tau2ptr = tau1tau2.ObtainReadData();
        Vector nu1 = etax.Field("nu1");
        Vector nu2 = etax.Field("nu2");
        const realdp *nu1TV = nu1.ObtainReadData();
        const realdp *nu2TV = nu2.ObtainReadData();
        
        HasHHR = false; HInvTran(x, etax, y, Hx, start, end, result); HasHHR = true;
        
        realdp *resultTV = result->ObtainWritePartialData();
        char *sider = const_cast<char *> ("r");
        integer ell = Hx.Getsize()[0], length = etax.Getlength();
        realdp *work = new realdp[ell];

        /* resultTV(:, start : start + length - 1) <- resultTV(:, start : start + length - 1) * (I - tau1tau2(0) * nu1TV * nu1TV^T),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sider, &ell, &length, const_cast<realdp *> (nu1TV), const_cast<realdp *> (tau1tau2ptr), resultTV + start * ell, &ell, work);
        /* resultTV(:, start : start + length - 1) <- resultTV(:, start : start + length - 1) * (I - tau1tau2(1) * nu2TV * nu2TV^T),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sider, &ell, &length, const_cast<realdp *> (nu2TV), const_cast<realdp *> (tau1tau2ptr + 1), resultTV + start * ell, &ell, work);
        delete[] work;
        return *result;
    };

    LinearOPE &Manifold::LCTranH(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, integer start, integer end, LinearOPE *result) const
    {
        if (!etax.FieldsExist("tau1tau2"))
        {
            Obtainnu1nu2forLC(x, etax, y);
        }
        Element tau1tau2 = etax.Field("tau1tau2");
        const realdp *tau1tau2ptr = tau1tau2.ObtainReadData();
        Vector nu1 = etax.Field("nu1");
        Vector nu2 = etax.Field("nu2");
        const realdp *nu1TV = nu1.ObtainReadData();
        const realdp *nu2TV = nu2.ObtainReadData();
        HasHHR = false; TranH(x, etax, y, Hx, start, end, result); HasHHR = true;
        realdp *resultTV = result->ObtainWritePartialData();
        
        char *sidel = const_cast<char *> ("l");
        integer ell = Hx.Getsize()[0], length = etax.Getlength();
        realdp *work = new realdp[ell];
        /* resultTV(start : start + length - 1, :) <- (I - tau1tau2(0) * nu1TV * nu1TV^T) * resultTV(start : start + length - 1, :),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sidel, &length, &ell, const_cast<realdp *> (nu1TV), const_cast<realdp *> (tau1tau2ptr), resultTV + start, &ell, work);
        /* resultTV(start : start + length - 1, :) <- (I - tau1tau2(1) * nu2TV * nu2TV^T) * resultTV(start : start + length - 1, :),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sidel, &length, &ell, const_cast<realdp *> (nu2TV), const_cast<realdp *> (tau1tau2ptr + 1), resultTV + start, &ell, work);
        delete[] work;
        return *result;
    };

    LinearOPE &Manifold::LCTranHInvTran(const Variable &x, const Vector &etax, const Variable &y, const LinearOPE &Hx, LinearOPE *result) const
    {
        if (!etax.FieldsExist("tau1tau2"))
        {
            Obtainnu1nu2forLC(x, etax, y);
        }
        
        Element tau1tau2 = etax.Field("tau1tau2");
        const realdp *tau1tau2ptr = tau1tau2.ObtainReadData();
        Vector nu1 = etax.Field("nu1");
        Vector nu2 = etax.Field("nu2");
        const realdp *nu1TV = nu1.ObtainReadData();
        const realdp *nu2TV = nu2.ObtainReadData();
        HasHHR = false; TranHInvTran(x, etax, y, Hx, result); HasHHR = true;
        realdp *resultTV = result->ObtainWritePartialData();
        
        char *sidel = const_cast<char *> ("l"), *sider = const_cast<char *> ("r");
        integer ell = Hx.Getsize()[0], length = etax.Getlength();
        realdp *work = new realdp[ell];
        /* resultTV <- resultTV * (I - tau1tau2(0) * nu1TV * nu1TV^T),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sider, &ell, &length, const_cast<realdp *> (nu1TV), const_cast<realdp *> (tau1tau2ptr), resultTV, &ell, work);
        /* resultTV <- resultTV * (I - tau1tau2(1) * nu2TV * nu2TV^T),
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sider, &ell, &length, const_cast<realdp *> (nu2TV), const_cast<realdp *> (tau1tau2ptr + 1), resultTV, &ell, work);
        /* resultTV <- (I - tau1tau2(0) * nu1TV * nu1TV^T) * resultTV,
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sidel, &length, &ell, const_cast<realdp *> (nu1TV), const_cast<realdp *> (tau1tau2ptr), resultTV, &ell, work);
        /* resultTV <- (I - tau1tau2(1) * nu2TV * nu2TV^T) * resultTV,
        details: www.netlib.org/lapack/explore-html/db/d10/larfx_8f.html */
        larfx_(sidel, &length, &ell, const_cast<realdp *> (nu2TV), const_cast<realdp *> (tau1tau2ptr + 1), resultTV, &ell, work);
        delete[] work;
        return *result;
    };

//    void Manifold::SetStartVectorTransportSmooth(void) const
//    {
//        IsVectTranSmooth = true;
//    };

    void Manifold::CheckEWGLdW(const Problem *prob, Variable x, Variable SMz) const
    {
//        Variable x = prob->GetDomain()->RandominManifold();
//        integer dimNorVec = prob->GetDomain()->GetExtrDim() - prob->GetDomain()->GetIntrDim();
        Vector Fz(EMPTYNORINTR), SMd(EMPTYNORINTR), Fd(EMPTYNORINTR), JSMd(EMPTYNORINTR); //SMz(EMPTYNORINTR),
//        SMz.RandGaussian();
        SMd.RandGaussian();
        SMd = SMd / SMd.Fnorm();

        Vector Weight;
        prob->PreConditioner(x, Weight, &Weight);

        Vector etax(prob->GetDomain()->GetEMPTYEXTR());
        etax.RandGaussian();
        prob->GetDomain()->ExtrProjection(x, etax, &etax);
        
        Vector BLambda(x), DLambda(x);
        
        ComputeBLambda(x, SMz, Weight, etax, prob, &BLambda);
        EW(x, BLambda, Weight, prob, &DLambda, &Fz);
        
        GLdW(x, SMd, Weight, BLambda, prob, &JSMd);
        
        ComputeBLambda(x, SMz + 1e-5 * SMd, Weight, etax, prob, &BLambda);
        EW(x, BLambda, Weight, prob, &DLambda, &Fd);
        
        std::cout << "norm((Fd - Fz) * 1e5 - JSMd):" << ((Fd - Fz) * 1e5 - JSMd).Fnorm() << std::endl;
    //    ((Fd - Fz) * 1e5 - JSMd).Print("norm(FD GLdW - GLdW):");
        
    //    ((Fd - Fz) * 1e5).Print("FD GLdW:");
    //    JSMd.Print("GLdW:");
    };

//    Vector &Manifold::TangentSpaceProximalMap(Variable &x, const Vector &etax, realdp adavalue, realdp SMtol, realdp SMlambda, const Problem *prob, Vector *inoutinitD, integer *outSMiter, integer *outSMCGiter, Vector *result) const
//    { /*eta = argmin g(gfx, eta) + 0.5 \|eta\|_W^2 + h(eta), W is either a scalar or a weight matrix that is given by function "PreConditioner" defined in Problem.
//              Only extrinsic representation is supported. */
//        /* Use LBFGS rather than Newton-CG */
//
//        integer dimNorVec = ExtrinsicDim - IntrinsicDim;
//        if(inoutinitD->GetSpace() == nullptr)
//        {
//            *inoutinitD = Vector (EMPTYNORINTR);
//            inoutinitD->SetToZeros();
//        }
//        Vector SMz(EMPTYNORINTR), Fz(EMPTYNORINTR), SMd(EMPTYNORINTR), SMu(EMPTYNORINTR), SMv(EMPTYNORINTR), Fu(EMPTYNORINTR), Fv(EMPTYNORINTR);
//
//        /*parameters*/
//        realdp SMm = 4, SMalpha = 0.01, SMgamma = 1;
//        integer SMmaxiter = 500, maxSMbtiter = 10;
//
//        /* for the algorithm */
//        integer SMbtiter = 0, SMCurrentlength = 0;
//        Vector Weight;
//        prob->PreConditioner(x, Weight, &Weight);
//        Weight.ScalarTimesThis(adavalue); /*Weight = adavalue * Weight*/
//
//        Vector S(dimNorVec, SMm), Y(dimNorVec, SMm); //, Psi(dimNorVec, SMm * 2), V(SMm * 2, SMm * 2);
//        S.NewMemoryOnWrite(); Y.NewMemoryOnWrite(); //-- Psi.NewMemoryOnWrite(); V.NewMemoryOnWrite();
//        realdp *Sptr = S.ObtainWriteEntireData(), *Yptr = Y.ObtainWriteEntireData(); //--, *Psiptr = Psi.ObtainWriteEntireData(), *Vptr = V.ObtainWriteEntireData();
//
//        realdp nFz = 0, nFu = 0, mu = 0;
//        SMz = *inoutinitD;
//        Vector BLambda(x);
//        ComputeBLambda(x, SMz, Weight, etax, prob, &BLambda);
//        EW(x, BLambda, Weight, prob, &Fz);
//        nFz = Fz.Fnorm();
//
//        SMd.SetToZeros();
//
//        (*outSMiter) = 0;
//        (*outSMCGiter) = 0;
////        SMtol = 1e-10;//---
//        while(nFz * nFz > SMtol && (*outSMiter) < SMmaxiter)
//        {
//            /* find the search direction, the search direction is in SMd */
//            mu = (nFz * nFz < 0.01 ? nFz * nFz : 0.01);
//            if(SMCurrentlength != 0)
//            {
//                Vector calD = S.GetSubmatrix(0, dimNorVec - 1, 0, SMCurrentlength - 1), calF = Y.GetSubmatrix(0, dimNorVec - 1, 0, SMCurrentlength - 1);
//                Vector calC(dimNorVec, SMCurrentlength * 2); calC.NewMemoryOnWrite();
//                calC.SubmatrixAssignment(0, dimNorVec - 1, 0, SMCurrentlength - 1, SMgamma * calD);
//                calC.SubmatrixAssignment(0, dimNorVec - 1, SMCurrentlength, 2 * SMCurrentlength - 1, calF);
//                Vector calV(2 * SMCurrentlength, 2 * SMCurrentlength); calV.SetToZeros();
//                calV.SubmatrixAssignment(0, SMCurrentlength - 1, 0, SMCurrentlength - 1, SMgamma * (calD.GetTranspose() * calD));
//                realdp *calVptr = calV.ObtainWritePartialData();
//                for(integer i = 0; i < SMCurrentlength; i++)
//                {
//                    for(integer j = 0; j < i; j++)
//                    {
//                        calVptr[i + 2 * SMCurrentlength * (j + SMCurrentlength)] = S.GetSubmatrix(0, dimNorVec - 1, i, i).DotProduct(Y.GetSubmatrix(0, dimNorVec - 1, j, j));
//                        calVptr[j + SMCurrentlength + 2 * SMCurrentlength * i] = calVptr[i + 2 * SMCurrentlength * (j + SMCurrentlength)];
//                    }
//                    calVptr[i + SMCurrentlength + 2 * SMCurrentlength * (i + SMCurrentlength)] = - S.GetSubmatrix(0, dimNorVec - 1, i, i).DotProduct(Y.GetSubmatrix(0, dimNorVec - 1, i, i));
//                }
//                Vector calR = calV - calC.GetTranspose() * calC / (SMgamma + mu);
//                SMd = (-1.0 / (SMgamma + mu)) * Fz - (1.0 / (SMgamma + mu) / (SMgamma + mu)) * calC * (calR % (calC.GetTranspose() * Fz));
//            } else
//                SMd = (-1.0 / (SMgamma + mu)) * Fz;
//
////            std::cout << "SMd Fz:" << SMd.DotProduct(Fz) << ", SMd norm:" << SMd.Fnorm() << ", Fz norm:" << Fz.Fnorm() << std::endl;//---
//
//            /* backtracking line search */
//            SMu = SMz; SMu.AlphaXaddThis(1, SMd); /* SMu = SMz + SMd; */
//            ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
//            EW(x, BLambda, Weight, prob, &Fu);
//            nFu = Fu.Fnorm();
//
//            /* a simple backtracking */
//            SMalpha = 1;
//            SMbtiter = 0;
//            while(0)//--nFu * nFu > nFz * nFz * (1 - 0.001 * SMalpha) && SMbtiter < maxSMbtiter)
//            {
//                SMalpha *= 0.5;
//                SMu = SMz; SMu.AlphaXaddThis(SMalpha, SMd); /* SMu = SMalpha * SMd + SMz; */
//                ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
//                EW(x, BLambda, Weight, prob, &Fu);
//                nFu = Fu.Fnorm();
//                SMbtiter++;
//            }
//            (*outSMCGiter) = (*outSMCGiter) + SMbtiter;
//
//            if(SMCurrentlength < SMm)
//            {
//                S.SubmatrixAssignment(0, dimNorVec - 1, SMCurrentlength, SMCurrentlength, SMu - SMz);
//                Y.SubmatrixAssignment(0, dimNorVec - 1, SMCurrentlength, SMCurrentlength, Fu - Fz);
//                SMCurrentlength++;
//            } else
//            {
//                for(integer i = 0; i < SMm - 1; i++)
//                {
//                    S.SubmatrixAssignment(0, dimNorVec - 1, i, i, S.GetSubmatrix(0, dimNorVec - 1, i + 1, i + 1));
//                    Y.SubmatrixAssignment(0, dimNorVec - 1, i, i, Y.GetSubmatrix(0, dimNorVec - 1, i + 1, i + 1));
//                }
//                if(SMm > 0)
//                {
//                    S.SubmatrixAssignment(0, dimNorVec - 1, SMm - 1, SMm - 1, SMu - SMz);
//                    Y.SubmatrixAssignment(0, dimNorVec - 1, SMm - 1, SMm - 1, Fu - Fz);
//                }
//            }
//            SMgamma = (Fu - Fz).DotProduct(Fu - Fz) / ((SMu - SMz).DotProduct(Fu - Fz));
//
////            std::cout << "SMiter:" << (*outSMiter) << ", nFz:" << nFz << ", SMgamma:" << SMgamma << ", SMalpha:" << SMalpha << ", yy:" << (Fu - Fz).DotProduct(Fu - Fz) << ",ss:" << ((SMu - SMz).DotProduct(SMu - SMz)) << std::endl;//--
//            SMz = SMu;
//            Fz = Fu;
//            nFz = nFu;
//            (*outSMiter)++;
//        }
//        std::cout << "Final SMiter:" << (*outSMiter) << ", nFz:" << nFz << std::endl;//---
//        *inoutinitD = SMz;
//        prob->ProxW(BLambda, Weight, result); result->AlphaXaddThis(-1, x); /*result = prob->ProxW(BLambda, Weight) - x*/
//        Projection(x, *result, result);
//        return *result;
//    };

//    Vector &Manifold::TangentSpaceProximalMap(Variable &x, const Vector &etax, realdp adavalue, realdp SMtol, realdp SMlambda, const Problem *prob, Vector *inoutinitD, integer *outSMiter, integer *outSMCGiter, Vector *result) const
//    { /*eta = argmin g(gfx, eta) + 0.5 \|eta\|_W^2 + h(eta), W is either a scalar or a weight matrix that is given by function "PreConditioner" defined in Problem.
//              Only extrinsic representation is supported. */
//        /* use line-search Newton-CG */
//        integer dimNorVec = ExtrinsicDim - IntrinsicDim;
//        if(inoutinitD->GetSpace() == nullptr)
//        {
//            *inoutinitD = Vector (EMPTYNORINTR);
//            inoutinitD->SetToZeros();
//        }
////        inoutinitD->SetToZeros();//----
//        Vector SMz(EMPTYNORINTR), Fz(EMPTYNORINTR), SMd(EMPTYNORINTR), SMu(EMPTYNORINTR), SMv(EMPTYNORINTR), Fu(EMPTYNORINTR), Fv(EMPTYNORINTR);
//
////        CheckEWGLdW(prob);//---
//
//        /*parameters*/
//        realdp SMtau = 0.1, SMalpha = 0.1, SMnu = 0.9, SMeta1 = 0.1, SMeta2 = 0.9, SMgamma1 = 1.01, SMgamma2 = 1.1, SMLipCon = 100;
//        realdp BBstepsize = 1;
//        integer SMmaxiter = 20, maxSMbtiter = 10; // SMmaxiter = 5000
//        std::string status;
//        integer SMbtiter = 0;
//        bool earlytermination = false;
//        Vector Weight;
//        prob->PreConditioner(x, Weight, &Weight);
//        Weight.ScalarTimesThis(adavalue); /*Weight = adavalue * Weight*/
//
//        realdp nFz = 0, nFu = 0, nFv = 0, mu = 0, xi = 0;
//        SMz = *inoutinitD;
//        Vector BLambda(x);
////        printf("time71:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
////        SMz.Print("SMz:");//---
//        ComputeBLambda(x, SMz, Weight, etax, prob, &BLambda);
////        printf("time72:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//
//        EW(x, BLambda, Weight, prob, &Fz);
////        printf("time73:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        nFz = Fz.Fnorm();
//        xi = nFz;
//
//        SMd.SetToZeros();
//
//        (*outSMiter) = 0;
//        (*outSMCGiter) = 0;
//        integer outSMCGiter_i = 0;
////        SMtol = 1e-10;
//        while(nFz * nFz > SMtol && (*outSMiter) < SMmaxiter)
//        {
////                 std::cout << "h4" << std::endl;//--
//            myCG(x, Fz, dimNorVec, (nFz * nFz < 0.01 ? nFz * nFz : 0.01), nFz  * (nFz < 0.1 ? nFz : 0.1), 0, static_cast<integer> (dimNorVec + 5), Weight, BLambda, SMd, prob, &outSMCGiter_i, &SMd);
////            SMd = -BBstepsize *  Fz;
////            printf("time76:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//            (*outSMCGiter) = (*outSMCGiter) + outSMCGiter_i;
//            SMu = SMz; SMu.AlphaXaddThis(1, SMd); /* SMu = SMz + SMd; */
////            std::cout << "SMu Fnorm:" << SMu.Fnorm() << std::endl;//---
//            ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
//            EW(x, BLambda, Weight, prob, &Fu);
//            nFu = Fu.Fnorm();
//
////            printf("time77:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//            /* a simple backtracking */
//            SMalpha = 1;
//            SMbtiter = 0;
////            while(nFu * nFu > nFz * nFz && SMbtiter < maxSMbtiter)
//            while(nFu * nFu > nFz * nFz * (1 - 0.001 * SMalpha) && SMbtiter < maxSMbtiter)
//            {
////                 std::cout << "h5" << std::endl;//--
//                SMalpha *= 0.5;
//                SMu = SMz; SMu.AlphaXaddThis(SMalpha, SMd); /* SMu = SMalpha * SMd + SMz; */
//                ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
//                EW(x, BLambda, Weight, prob, &Fu);
//                nFu = Fu.Fnorm();
//                SMbtiter++;
//            }
//            if(SMbtiter >= maxSMbtiter)
//            {
//                break;
//            }
////            printf("time78:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
////            BBstepsize = SMd.DotProduct(Fu - Fz) / (Fu - Fz).DotProduct(Fu - Fz);
////            BBstepsize = SMd.DotProduct(SMd) / SMd.DotProduct(Fu - Fz);
////            std::cout << "BB stepsize:" << (Fu - Fz).DotProduct(Fu - Fz) / SMd.DotProduct(Fu - Fz) << std::endl;//---
//            SMz = SMu;
//            Fz = Fu;
//            nFz = nFu;
////            CheckEWGLdW(prob, x, SMz);//---
//            std::cout << "SMiter:" << (*outSMiter) << ", nFz:" << nFz << ", SMalpha:" << SMalpha << std::endl;//---
//            (*outSMiter)++;
//        }
//        std::cout << "Final SMiter:" << (*outSMiter) << ", nFz:" << nFz << std::endl;//---
//        *inoutinitD = SMz;
//        prob->ProxW(BLambda, Weight, result); result->AlphaXaddThis(-1, x); /*result = prob->ProxW(BLambda, Weight) - x*/
//        Projection(x, *result, result);
//        return *result;
//    };
//
//    Vector &Manifold::myCG(const Variable &x, const Vector &nb, integer dimNorVec, realdp mu, realdp tol, realdp lambdanFz, integer maxiter, const Vector &Weight, const Vector &BLambda, const Vector &init, const Problem *prob, integer *CGiter, Vector *result) const
//    {
//        Vector r(EMPTYNORINTR), p(EMPTYNORINTR), Ap(EMPTYNORINTR), SMd(init);
//        realdp rr0, alpha, beta;
//
////        printf("time74:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        GLdW(x, SMd, Weight, BLambda, prob, &Ap).AlphaXaddThis(mu, SMd); /* Ap = mu * SMd + GLdW(x, SMd, Weight, BLambda, prob, &tmp); */
////        printf("time75:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
//        *result = SMd;
//        r = Ap; r.ScalarTimesThis(-1); r.AlphaXaddThis(-1, nb); /*r = -1 * nb - Ap; r is negative residual*/
//        p = r;
//        (*CGiter) = 0;
//        while(r.Fnorm() > tol && (*CGiter) < maxiter)
//        {
////                 std::cout << "h6" << std::endl;//--
//            GLdW(x, p, Weight, BLambda, prob, &Ap).AlphaXaddThis(mu, p); /* Ap = mu * p + GLdW(x, p, Weight, BLambda, prob, &tmp); */
//            rr0 = r.DotProduct(r);
//            alpha = rr0 / p.DotProduct(Ap);
//            result->AlphaXaddThis(alpha, p); /* (*result) = alpha * p + (*result); */
//            r.AlphaXaddThis(- alpha, Ap); /* r = r - alpha * Ap; */
//            beta = r.DotProduct(r) / rr0;
//            p.ScalarTimesThis(beta); p.AlphaXaddThis(1, r); /* p = r + beta * p; */
//            (*CGiter)++;
//        }
//        std::cout<< "\t CGiter:" << *CGiter << ", r Fnorm:" << r.Fnorm() << ", tol:" << tol << std::endl;//---
//
//        return *result;
//    };

    Vector &Manifold::TangentSpaceProximalMap(Variable &x, const Vector &etax, realdp SMtol, realdp SMlambda, integer ProxMaptype, integer iter, const Vector &Weight, const Problem *prob, Vector *inoutinitD, integer *outSMitertotal, integer *outSMCGitertotal, integer *PMiter, Vector *result) const
    { /* ProxMaptype: LSPG_GLOBAL (0), LSPG_UNILIMIT (1), LSPG_LOCAL (2) */
        *outSMitertotal = 0;
        *outSMCGitertotal = 0;
        integer outSMiter = 0, outSMCGiter = 0;
//        std::cout << "time1:" << static_cast<realdp>(getTickCount() - starttime) / CLK_PS << std::endl;//----
        TangentSpaceProximalMapSub(x, etax, SMtol, SMlambda, Weight, prob, inoutinitD, &outSMiter, &outSMCGiter, result);
        *outSMitertotal = outSMiter; *outSMCGitertotal = outSMCGiter;
//        std::cout << "time2:" << static_cast<realdp>(getTickCount() - starttime) / CLK_PS << std::endl;//----
        if(ProxMaptype == 0)
        { /* if only global convergence is needed, then it is not necessary to do more than one iteration. */
            return *result;
        }
        
        integer nblock = Weight.Getlength();
        integer blocksize = x.Getlength() / nblock;
        /*xix denotes the iterate, zetay denotes the T_R^{-\sharp} (etax + W xix), xiy denotes the search direction, zetax denotes T_R^{-1} xiy */
        Vector xix(*result), xix2(*result), zetay(*result), xiy(*result), zetax(*result), initialxix(*result);
        realdp ndir = 0;
        Variable y(x);
        realdp alpha = 0; /*stepsize*/
        realdp psi = 10000;
        realdp nxix = xix.Fnorm(), nzetax = 0;
        realdp f1 = ell(x, etax, xix, Weight, prob), f2 = 0;
        integer lsiter = 0, maxPMiter = 10, maxlsiter = 10;
        *PMiter = 0;
//        printf("PMiter:%d, ell:%.10e, nxix:%e\n", PMiter, f1, nxix);
//        std::cout << "time3:" << static_cast<realdp>(getTickCount() - starttime) / CLK_PS << std::endl;//----
//        Vector LWeight;
//        realdp norminverseTVestimation = 0;
        
//        Vector zeroxix(xix);//----
//        ScalarTimesVector(x, 0, xix, &zeroxix);//----
//        realdp f0 = ell(x, etax, zeroxix, Weight, prob); //---
//        realdp f10 = f1;//---

        Retraction(x, xix, &y);
        zetay = xix;
        if(nblock != 1)
        {
            realdp *zetayptr = zetay.ObtainWritePartialData();
            for(integer i = 0; i < nblock; i++)
            {
                realdp tmp = Weight.ObtainReadData()[i];
                scal_(&blocksize, &tmp, zetayptr + i * blocksize, &GLOBAL::IONE);
            }
        } else
        {
            ScalarTimesVector(x, Weight.ObtainReadData()[0], zetay, &zetay);
        }
        ScalarVectorAddVector(x, 1, etax, zetay, &zetay);
        
        InverseVecTranDiffRetAdjoint(x, xix, y, zetay, &zetay);
        
        while(*PMiter < maxPMiter)
        {
//            Retraction(x, xix, &y);
//            zetay = xix;
//            if(nblock != 1)
//            {
//                realdp *zetayptr = zetay.ObtainWritePartialData();
//                for(integer i = 0; i < nblock; i++)
//                {
//                    realdp tmp = Weight.ObtainReadData()[i];
//                    scal_(&blocksize, &tmp, zetayptr + i * blocksize, &GLOBAL::IONE);
//                }
//            } else
//            {
//                ScalarTimesVector(x, Weight.ObtainReadData()[0], zetay, &zetay);
//            }
//            ScalarVectorAddVector(x, 1, etax, zetay, &zetay);
//
//            InverseVecTranDiffRetAdjoint(x, xix, y, zetay, &zetay);
//            std::cout << "h1" << std::endl;//---
            TangentSpaceProximalMapSub(y, zetay, SMtol, SMlambda, Weight, prob, inoutinitD, &outSMiter, &outSMCGiter, &xiy);
            *outSMitertotal += outSMiter; *outSMCGitertotal += outSMCGiter;

            if(ProxMaptype == 1) /*LSPG_UNILIMIT*/
            {/*The coefficient of psi seems to be problem dependent.*/
                psi = 500.0 / std::pow(static_cast<realdp> (iter + 1), 2.02);
            } else
            if(ProxMaptype == 2) /*LSPG_LOCAL*/
            {
                realdp tmp1 = 500.0 / std::pow(static_cast<realdp> (iter + 1), 2.02), tmp2 = 100.0 * nxix * nxix;
                psi = (tmp1 < tmp2) ? tmp1 : tmp2;
            }
//            std::cout << "h2" << std::endl;//---
            
            ndir = xiy.Fnorm();
            
            alpha = 1;
//            std::cout << "time6:" << static_cast<realdp>(getTickCount() - starttime) / CLK_PS << std::endl;//----
//            std::cout << "norm xiy:" << xiy.Fnorm() << ":" << std::sqrt(Metric(y, xiy, xiy)) << std::endl;//---
//            std::cout << "ttt:" << xiy.Field("dU").Fnorm() << std::endl;//---
            InverseVecTranDiffRet(x, xix, y, xiy, &zetax);
//            Projection(x, zetax, &zetax);//---
//            xiy.Print("xiy");//---
//            Vector ttmp(xiy);//---
//            VecTranDiffRet(x, xix, y, zetax, &ttmp);//----
//            ttmp.Print("T circ T^{-1} xiy:");//---
            
//            xix.AlphaXaddThis(alpha, zetax);
//            std::cout << "norm xix1:" << xix.Fnorm() << std::endl;//---
//            std::cout << "norm zetax:" << zetax.Fnorm() << std::endl;//---
            ScalarVectorAddVector(x, alpha, zetax, xix, &xix2);
            nzetax = zetax.Fnorm();
//            std::cout << "norm xix2:" << xix.Fnorm() << std::endl;//---
            f2 = ell(x, etax, xix2, Weight, prob);
            lsiter = 0;
//            std::cout << "h3" << std::endl;//---
            while(f2 >= f1 - 1e-3 * alpha * nzetax * nzetax && lsiter < maxlsiter)
            {
//                std::cout << "h31" << std::endl;//---
                alpha = alpha / 2;
                ScalarVectorAddVector(x, alpha, zetax, xix, &xix2);
//                xix.AlphaXaddThis(alpha, zetax);
                f2 = ell(x, etax, xix2, Weight, prob);
                lsiter++;
            }
//            std::cout << "PMiter:" << *PMiter << "f2:" << f2 << ", f1:" << f1 << ", :" << f1 - 1e-3 * alpha * nzetax * nzetax << ", alpha:" << alpha << "," << f1 - f2 << std::endl;//---
//            std::cout << "time7:" << static_cast<realdp>(getTickCount() - starttime) / CLK_PS << ", " << alpha << std::endl;//----

            if(lsiter == maxlsiter)
            {
//                printf("warning in TangentSpaceProximalMap: line search reaches the maximum number iterations!\n");
                SMtol *= 1e-3;
                if(SMtol <= 1e-10)
                    break;
                else
                    continue;
            }
            
            xix = xix2;
            nxix = xix.Fnorm();
            (*PMiter)++;
//            std::cout << "h32" << std::endl;//---
            
//            std::cout << "ndir:" << ndir << ", psi:" << psi << std::endl;//---
            if(ndir <= psi)
                break;
//                std::cout << "h33" << std::endl;//---
            f1 = f2;
            
//            std::cout << "h34" << std::endl;//---
            Retraction(x, xix, &y);
            zetay = xix;
            if(nblock != 1)
            {
                realdp *zetayptr = zetay.ObtainWritePartialData();
                for(integer i = 0; i < nblock; i++)
                {
                    realdp tmp = Weight.ObtainReadData()[i];
                    scal_(&blocksize, &tmp, zetayptr + i * blocksize, &GLOBAL::IONE);
                }
            } else
            {
                ScalarTimesVector(x, Weight.ObtainReadData()[0], zetay, &zetay);
            }
//            std::cout << "h4" << std::endl;//---
            ScalarVectorAddVector(x, 1, etax, zetay, &zetay);
            
            InverseVecTranDiffRetAdjoint(x, xix, y, zetay, &zetay);
//            std::cout << "time8:" << static_cast<realdp>(getTickCo-unt() - starttime) / CLK_PS << std::endl;//----
//            printf("PMiter:%d, ell:%.10e, nxix:%e, ndir:%e, psi:%e \n", PMiter, f2, nxix, ndir, psi);
        }
//        printf("PMiter:%d, ell:%.10e, nxix:%e, ndir:%e, psi:%e \n", PMiter, f2, nxix, ndir, psi);
        if(*PMiter == maxPMiter)
        {
            printf("warning in TangentSpaceProximalMap: the maximum number iterations!\n");
//            xix = initialxix;
        }

//        std::cout << "test:" << (f10 - ell(x, etax, xix, Weight, prob)) / (f0 - ell(x, etax, xix, Weight, prob)) << std::endl;//---
        
//        std::cout << "h41" << std::endl;//---
        *result = xix;
        return *result;
    };

    realdp Manifold::ell(const Variable &x, const Vector &etax, const Vector &xix, const Vector &Weight, const Problem *prob) const
    {/* compute: ell(xix) = <xix, etax> + 1/2 * <xix, W xix> + g(R_x(xix))*/
        Vector zetax = xix;
        integer nblock = Weight.Getlength();
        if(nblock != 1)
        {
            realdp *zetaxptr = zetax.ObtainWritePartialData();
            integer blocksize = x.Getlength() / nblock;
            for(integer i = 0; i < nblock; i++)
            {
                const realdp tmp = Weight.ObtainReadData()[i];
                scal_(&blocksize, const_cast<realdp *> (&tmp), zetaxptr + i * blocksize, &GLOBAL::IONE);
            }
        } else
        {
            ScalarTimesVector(x, Weight.ObtainReadData()[0], zetax, &zetax);
        }
        Variable y(x);
        Retraction(x, xix, &y);
//        ExtrProjection(x, zetax, &zetax);
//        std::cout << "h1:" << Metric(x, etax, xix) << " : " << Metric(x, xix, zetax) / 2 << " : " << prob->g(y) << std::endl;//---
        return Metric(x, etax, xix) + Metric(x, xix, zetax) / 2 + prob->g(y);
    };

    Vector &Manifold::TangentSpaceProximalMapSub(Variable &x, const Vector &etax, realdp SMtol, realdp SMlambda, const Vector &Weight, const Problem *prob, Vector *inoutinitD, integer *outSMiter, integer *outSMCGiter, Vector *result) const
    { /* result = \argmin_{p \in \T_x M} <p, etax> + 0.5 <p, Weight p> + h(x+p), W is either a scalar or a weight matrix that is given by function "PreConditioner" defined in Problem.
              Only extrinsic representation is supported. */
        /* use Newton-CG with projection steps */
        integer dimNorVec = ExtrinsicDim - IntrinsicDim;
        if(inoutinitD->GetSpace() == nullptr)
        {
            *inoutinitD = Vector (EMPTYNORINTR);
            inoutinitD->SetToZeros();
        }
//        inoutinitD->SetToZeros();//----
        Vector SMz(EMPTYNORINTR), Fz(EMPTYNORINTR), SMd(EMPTYNORINTR), SMu(EMPTYNORINTR), SMv(EMPTYNORINTR), Fu(EMPTYNORINTR), Fv(EMPTYNORINTR);

//        CheckEWGLdW(prob);//---

        /*parameters*/
        realdp SMtau = 0.1, SMalpha = 0.1, SMeta1 = 0.1, SMeta2 = 0.9, SMgamma1 = 3, SMgamma2 = 10, SMLipCon = 10; //, SMnu = 0.99
        integer SMmaxiter = 20, maxSMbtiter = 5;
        std::string status;
        integer SMbtiter = 0;
//        bool earlytermination = false;

        realdp nFz = 0, nFu = 0, nFv = 0, mu = 0, xi = 0;
        SMz = *inoutinitD;
        Vector BLambda(x), DLambda(x);
        ComputeBLambda(x, SMz, Weight, etax, prob, &BLambda);

        EW(x, BLambda, Weight, prob, &DLambda, &Fz);
        nFz = Fz.Fnorm();
        xi = nFz;

        Projection(x, DLambda, &DLambda);
        realdp phieta = std::pow(Metric(x, DLambda, DLambda), 0.25);

        SMd.SetToZeros();

        (*outSMiter) = 0;
        (*outSMCGiter) = 0;
        integer outSMCGiter_i = 0;
//        realdp phieta = 0;
//        while((nFz * nFz > SMtol && (*outSMiter) < SMmaxiter))//--- || (*outSMiter) < 1 )
//        std::cout << "nFz:" << nFz << ", " << 10 * phieta << std::endl;//----
        while((nFz > ((0.5 * SMtol < 10 * phieta) ? 0.5 * SMtol : 10 * phieta) && (*outSMiter) < SMmaxiter))//--- || (*outSMiter) < 1 )
        {
//            std::cout << "h5" << std::endl;//---
            mu = ((nFz < 0.1) ? nFz : 0.1);
            mu = ((mu > 1e-11) ? mu : 1e-11);
            mu = SMlambda *  mu;
//            mu = 1e-5;//---
//            std::cout << "SMlambda:" << SMlambda << ", mu:" << mu << std::endl;//---
//            mu = 10; //--- 1e-5;//------
//            myCG(x, Fz, dimNorVec, mu, SMtau, SMlambda * nFz, ((dimNorVec < 30) ? dimNorVec : 30), Weight, BLambda, SMd, prob, &outSMCGiter_i, &SMd);
            myCG(x, Fz, dimNorVec, mu, SMtau, SMlambda * nFz, (dimNorVec > 10) ? 10 : dimNorVec, Weight, BLambda, SMd, prob, &outSMCGiter_i, &SMd);
//            std::cout << "h6" << std::endl;//---
            (*outSMCGiter) = (*outSMCGiter) + outSMCGiter_i;
            SMu = SMz; SMu.AlphaXaddThis(1, SMd); /* SMu = SMz + SMd; */
            ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
            EW(x, BLambda, Weight, prob, &DLambda, &Fu);
            nFu = Fu.Fnorm();

            /* approach in [XLWZ2017]: A Regularized Semi-Smooth Newton Method with
            Projection Steps for Composite Convex Programs */
//            if(nFu <= SMnu * xi)
//            { //Newton step is successful
//                SMz = SMu;
//                Fz = Fu;
//                nFz = nFu;
//                xi = nFu;
//                SMlambda *= 0.25;
//                SMlambda = (SMlambda < 1e-5) ? 1e-5 : SMlambda;
//                std::cout << "Newton step" << std::endl;//---

            SMalpha = 1;
            SMbtiter = 0;
            while(nFu * nFu > nFz * nFz * (1 - 0.001 * SMalpha) && SMbtiter < maxSMbtiter)
            {
                SMalpha *= 0.5;
                SMu = SMz; SMu.AlphaXaddThis(SMalpha, SMd); /* SMu = SMalpha * SMd + SMz; */
                ComputeBLambda(x, SMu, Weight, etax, prob, &BLambda);
                EW(x, BLambda, Weight, prob, &DLambda, &Fu);
                nFu = Fu.Fnorm();
                SMbtiter++;
            }
            if(SMbtiter < maxSMbtiter)
            {
                SMz = SMu;
                Fz = Fu;
                nFz = nFu;
            }
            if(SMbtiter >= maxSMbtiter)
            {
                SMv = SMz - (Fu.DotProduct(SMz - SMu) / Fu.DotProduct(Fu)) * Fu;
                ComputeBLambda(x, SMv, Weight, etax, prob, &BLambda);
                EW(x, BLambda, Weight, prob, &DLambda, &Fv);
                nFv = Fv.Fnorm();
                realdp rho = - SMd.DotProduct(Fu) / SMd.DotProduct(SMd);
//                std::cout << "rho:" << - SMd.DotProduct(Fu) / SMd.DotProduct(SMd) << std::endl;//---
                if(rho >= SMeta1)
                {
                    if(nFv <= nFz)
                    {
                        SMz = SMv;
                        Fz = Fv;
                        nFz = nFv;
//                        std::cout << "Projection step" << std::endl;//---
                    } else
                    {
                        SMz = SMz - (0.25 / SMLipCon) * Fz;
                        ComputeBLambda(x, SMz, Weight, etax, prob, &BLambda);
                        EW(x, BLambda, Weight, prob, &DLambda, &Fz);
                        nFz = Fz.Fnorm();
//                        std::cout << "Fixed-point step" << std::endl;//---
                    }
                }
                else
                {
//                    printf("Warning: Unsuccessful step in semi-smooth Newton method\n");
                }
                if(rho >= SMeta2)
                {
                    SMlambda *= 0.25;
                    SMlambda = (SMlambda < 1e-5) ? 1e-5 : SMlambda;
                } else
                if(rho >= SMeta1)
                {
                    SMlambda *= SMgamma1;
                } else
                {
                    SMlambda *= SMgamma2;
                }
            }

            Projection(x, DLambda, &DLambda);
            phieta = std::pow(Metric(x, DLambda, DLambda), 0.25);
//            std::cout << "SMtol:" << SMtol << ", phi:" << std::min(0.5, phieta) << std::endl;//---
            (*outSMiter)++;
//            std::cout << "SMiter:" << (*outSMiter) << ", nFz:" << nFz << ", mu:" << mu << ", SMlambda:" << SMlambda << ", SMalpha:" << SMalpha << std::endl;//---
        }
//        std::cout << "Final SMiter:" << (*outSMiter) << ", nFz:" << nFz << std::endl;//---
        *inoutinitD = SMz;
//        prob->ProxW(BLambda, Weight, result); result->AlphaXaddThis(-1, x); /*result = prob->ProxW(BLambda, Weight) - x*/
//        Projection(x, *result, result);
        Projection(x, DLambda, result);
        return *result;
    };

    Vector &Manifold::myCG(const Variable &x, const Vector &nb, integer dimNorVec, realdp mu, realdp tau, realdp lambdanFz, integer maxiter, const Vector &Weight, const Vector &BLambda, const Vector &init, const Problem *prob, integer *CGiter, Vector *result) const
    {
        Vector r(EMPTYNORINTR), p(EMPTYNORINTR), Ap(EMPTYNORINTR), SMd(init);
        realdp rr0, alpha, beta;

        GLdW(x, SMd, Weight, BLambda, prob, &Ap).AlphaXaddThis(mu, SMd); /* Ap = mu * SMd + GLdW(x, SMd, Weight, BLambda, prob, &tmp); */
        *result = SMd;
        r = Ap; r.ScalarTimesThis(-1); r.AlphaXaddThis(-1, nb); /*r = -1 * nb - Ap; r is negative residual*/
        p = r;
        (*CGiter) = 0;
        while(r.Fnorm() > tau * ((lambdanFz * result->Fnorm() < 1.0) ? lambdanFz * result->Fnorm() : 1.0) && (*CGiter) < maxiter)
        {
            GLdW(x, p, Weight, BLambda, prob, &Ap).AlphaXaddThis(mu, p); /* Ap = mu * p + GLdW(x, p, Weight, BLambda, prob, &tmp); */
            rr0 = r.DotProduct(r);
            alpha = rr0 / p.DotProduct(Ap);
            result->AlphaXaddThis(alpha, p); /* (*result) = alpha * p + (*result); */
            r.AlphaXaddThis(- alpha, Ap); /* r = r - alpha * Ap; */
            beta = r.DotProduct(r) / rr0;
            p.ScalarTimesThis(beta); p.AlphaXaddThis(1, r); /* p = r + beta * p; */
            (*CGiter)++;
        }
//        std::cout<< "\t CGiter:" << *CGiter << ", r Fnorm:" << r.Fnorm() << ", tau:" << tau << ", lambdanFz * result->Fnorm():" << lambdanFz * result->Fnorm() << std::endl;//---
        return *result;
    };

    Vector &Manifold::ComputeBLambda(const Variable &x, const Vector &d, const Vector &Weight, const Vector &gfx, const Problem *prob, Vector *result) const
    { /*B(Lambda) = x - W^{-1} ( \grad f(x) - calA^sharp * Lambda ), calA^sharp = B_x */
        calAadj(x, d, prob, result);
//        std::cout << result->Getlength() << ":" << gfx.Getlength() << std::endl;//---
        result->ScalarTimesThis(-1); result->AlphaXaddThis(1, gfx); /*result = gfx - calAadj(x, d, prob) */
        
        integer nblock = Weight.Getlength();
        integer blocksize = x.Getlength() / nblock;
        realdp *resultptr = result->ObtainWritePartialData();
        
        for(integer i = 0; i < nblock; i++)
        {
            const realdp L = Weight.ObtainReadData()[i];
            realdp tmp = 1.0 / L;
            scal_(&blocksize, &tmp, resultptr + i * blocksize, &GLOBAL::IONE);
        }
        result->ScalarTimesThis(-1); result->AlphaXaddThis(1, x); /* result = x - result; */
        return *result;
    };

    Vector &Manifold::EW(const Variable &x, const Vector &BLambda, const Vector &Weight, const Problem *prob, Vector *DLambda, Vector *result) const
    { /*Psi(Lambda) = calA( ProxW( BLambda ) - x ) */
//        Vector DLambda(BLambda);
        prob->ProxW(BLambda, Weight, DLambda);
        DLambda->AlphaXaddThis(-1, x); /* DLambda = prob->ProxW(BLambda, Weight) - x; */
        return calA(x, *DLambda, prob, result);
    };

    Vector &Manifold::GLdW(const Variable &x, const Vector &d, const Vector &Weight, const Vector &BLambda, const Problem *prob, Vector *result) const
    { /* J_{\Psi}(Lambda)[d] */
        Vector VecTMP1(x), VecTMP2(x);
//        printf("time741:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        calAadj(x, d, prob, &VecTMP1);
        
        realdp *Vec1ptr = VecTMP1.ObtainWritePartialData();
        integer nblock = Weight.Getlength();
        integer blocksize = x.Getlength() / nblock;
        for(integer i = 0; i < nblock; i++)
        {
            const realdp L = Weight.ObtainReadData()[i];
            realdp tmp = 1.0 / L;
            scal_(&blocksize, &tmp, Vec1ptr + i * blocksize, &GLOBAL::IONE);
        }
        
//        printf("time742:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        prob->CalJW(BLambda, VecTMP1, Weight, &VecTMP2);
//        printf("time743:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        
//        printf("time744:%f\n", static_cast<realdp>(getTickCount() - starttime) / CLK_PS);
        return calA(x, VecTMP2, prob, result);
    };

    Vector &Manifold::calA(const Variable &x, const Vector &DLambda, const Problem *prob, Vector *result) const
    {
        return ObtainNorVerIntr(x, DLambda, result);
        /*for debug*/
        /*testCalACalAadj(x, DLambda, ELambda); */
    };

    Vector &Manifold::calAadj(const Variable &x, const Vector &d, const Problem *prob, Vector *result) const
    {
        return ObtainNorVerExtr(x, d, result);
    };

/*
    void RPG::testCalACalAadj(Variable *x, Vector *DLambda, realdp *ELambda)
    {
        Prob->GetDomain()->ObtainNorVerIntr(x, DLambda, ELambda);
        realdp *tmpp = new realdp[dimNorVec];
        for(integer i = 0; i < dimNorVec; i++)
            tmpp[i] = genrandnormal();
        std::cout << "<A d, t>:" << dot_(&dimNorVec, ELambda, &GLOBAL::IONE, tmpp, &GLOBAL::IONE) << std::endl;

        Vector *Vtmp = DLambda->ConstructEmpty();
        calAadj(x, tmpp, Vtmp);
        std::cout << "<d, A^* t>:" << Prob->GetDomain()->Metric(nullptr, Vtmp, DLambda) << std::endl;
        delete Vtmp;
        delete[] tmpp;
    }
*/
}; /*end of ROPTLIB namespace*/
