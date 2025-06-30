
#include "Manifolds/Grassmann.h"

/*Define the namespace*/
namespace ROPTLIB{

	Grassmann::Grassmann(integer inn, integer inp)
	{
		HasHHR = false;

		IsIntrApproach = true;
//        IsVectTranSmooth = true;

		n = inn;
		p = inp;
		ExtrinsicDim = n * p;
		IntrinsicDim = (n - p) * p;
		name.assign("Grassmann");
        EMPTYEXTR = Vector (n, p);
        EMPTYINTR = Vector (IntrinsicDim);
        EMPTYNORINTR = Vector (p, p);
	};

    void Grassmann::ChooseParamsSet1(void)
    {
        IsIntrApproach = true;
    };

    void Grassmann::ChooseParamsSet2(void)
    {
        IsIntrApproach = false;
    };

	Grassmann::~Grassmann(void)
	{
	};

    Variable Grassmann::RandominManifold(void) const
    {
        Variable result(n, p);
        result.RandGaussian();
        result.QRDecom();
        return result.Field("_Q");
    };

	void Grassmann::CheckParams(void) const
	{
		Manifold::CheckParams();
		printf("%s PARAMETERS:\n", name.c_str());
		printf("n             :%15d,\t", n);
		printf("p             :%15d\n", p);
	};

	Vector &Grassmann::ExtrProjection(const Variable &x, const Vector &etax, Variable *result) const
	{
        Vector xTetax(p, p);
        xTetax.AlphaABaddBetaThis(1, x, GLOBAL::T, etax, GLOBAL::N, 0); /* xTetax = x^T * etax*/
        *result = etax;
        result->AlphaABaddBetaThis(-1, x, GLOBAL::N, xTetax, GLOBAL::N, 1); /* result = etax - x * x^T * etax */
        return *result;
	};

	Variable &Grassmann::Retraction(const Variable &x, const Vector &etax, Variable *result) const
	{
        /*polar retraction*/
        Vector exetax(EMPTYEXTR);
        if(IsIntrApproach)
            ObtainExtr(x, etax, &exetax);
        else
            exetax = etax;

        Vector xaddetax = x + exetax;
        xaddetax.QRDecom();
        Vector R = xaddetax.Field("_R");
        
        R.SVDDecom();
        
        *result = xaddetax.Field("_Q") * R.Field("_U") * R.Field("_Vt");

        result->AddToFields("_HHR", xaddetax.Field("_HHR"));
        result->AddToFields("_tau", xaddetax.Field("_tau"));
        result->AddToFields("_Vt", R.Field("_Vt"));
        result->AddToFields("_S", R.Field("_S"));

        return *result;
        
//        /* qf retraction: result = R_x(etax) = qf(x + etax) */
//        Vector exetax(EMPTYEXTR);
//        if(IsIntrApproach)
//            ObtainExtr(x, etax, &exetax);
//        else
//            exetax = etax;
//
//        Vector xaddetax = x + exetax;
//        xaddetax.QRDecom();
//        Vector R = xaddetax.Field("_R");
//        *result = xaddetax.Field("_Q");
//
//        realdp *resultptr = result->ObtainWritePartialData();
//        const realdp *Rptr = R.ObtainReadData();
//        for(integer i = 0; i < p; i++)
//        {
//            if(Rptr[i + i * p] < 0)
//            {
//                scal_(&n, &GLOBAL::DNONE, resultptr + i * n, &GLOBAL::IONE);
//            }
//        }
//        result->AddToFields("_HHR", xaddetax.Field("_HHR"));
//        result->AddToFields("_tau", xaddetax.Field("_tau"));
//
//        return *result;
	};

    Vector &Grassmann::InvRetraction(const Variable &x, const Variable &y, Vector *result) const
    {
        Vector A(p, p); A.AlphaABaddBetaThis(1, x, GLOBAL::T, y, GLOBAL::N, 0); /* A = x.GetTranspose() * y; */
        
        Vector tIp(p, p); tIp.SetToIdentity(); tIp = 2 * tIp;
        Vector soln = tIp.SYL(A, A.GetTranspose());
        Vector tmp(x);
        tmp.AlphaABaddBetaThis(1, y, GLOBAL::N, soln, GLOBAL::N, -1); /* y * tIp.SYL(A, A.GetTranspose()) - x; */
        
        if(IsIntrApproach)
        {
            return ObtainIntr(x, tmp, result);
        }
        
        return *result;
    };

	Vector &Grassmann::VecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result, bool IsEtaXiSameDir) const
	{
        if (IsEtaXiSameDir)
        {
            Vector etaxtmp(EMPTYEXTR);
            if(IsIntrApproach)
                ObtainExtr(x, etax, &etaxtmp);
            else
                etaxtmp = etax;
            realdp alpha = sqrt(Metric(x, xix, xix) / Metric(x, etax, etax));
            Vector Vt = y.Field("_Vt");
            Vector S = y.Field("_S");
            Vector tmp(EMPTYEXTR);
            tmp.AlphaABaddBetaThis(1, etaxtmp, GLOBAL::N, Vt, GLOBAL::T, 0); /*tmp = etaxtmp * Vt.GetTranspose();*/
            S = 1 / S;
            tmp = S.GetDiagTimesM(tmp, GLOBAL::R) * Vt;
            Vector tmp2(p, p); tmp2.AlphaABaddBetaThis(1, tmp, GLOBAL::T, tmp, GLOBAL::N, 0); /*tmp2 = tmp.GetTranspose() * tmp*/
            Vector exresult(tmp); exresult.AlphaABaddBetaThis(-alpha, y, GLOBAL::N, tmp2, GLOBAL::N, alpha); /* exresult = (tmp - y * (tmp.GetTranspose() * tmp)) * alpha; */

            if(IsIntrApproach)
                ObtainIntr(y, exresult, result);
            else
            {
                *result = exresult;
                ExtrProjection(y, *result, result);
            }

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
                    etax.AddToFields("betaTReta", (*result) * (betaptr[0] * EtatoXi));
                }
            }
            return *result;
        }
        
        Vector Vt = y.Field("_Vt"), S = y.Field("_S");
        Vector Vtmp(Vt); S.DiagTimesM(Vtmp); Vtmp = Vt.GetTranspose() * Vtmp;
        Vector Vtmp2 = xix / Vtmp;
//        Vector Vtmp3 = y.GetTranspose() * xix - xix.GetTranspose() * y;
//        Vector Omega = Vtmp3.SYL(Vtmp, Vtmp);
        Vector exresult(EMPTYEXTR);
        exresult = Vtmp2 - y * (y.GetTranspose() * Vtmp2);
        if(!IsIntrApproach)
        {
            *result = exresult;
            return *result;
        }
        
        ObtainIntr(y, exresult, result);

//        ExtrProjection(y, *result, result);
        return *result;
        
//        /* vector transport by differentiated the qf retraction */
//        realdp nxix = std::sqrt(Metric(x, xix, xix));
//
//        Vector exxix(EMPTYEXTR);
//        if(IsIntrApproach)
//            ObtainExtr(x, xix, &exxix);
//        else
//            exxix = xix;
//
//        Vector HHR = y.Field("_HHR");
//        const realdp *HHRptr = HHR.ObtainReadData();
//        realdp *exxixptr = exxix.ObtainWritePartialData();
//        trsm_(GLOBAL::R, GLOBAL::U, GLOBAL::N, GLOBAL::N, &n, &p, &GLOBAL::DONE, const_cast<realdp *> (HHRptr), &n, exxixptr, &n);
//        for(integer i = 0; i < p; i++)
//        {
//            if(HHRptr[i + i * n] < 0)
//                scal_(&n, &GLOBAL::DNONE, exxixptr + i * n, &GLOBAL::IONE);
//        }
//        Vector YtVRinv(p, p); YtVRinv.AlphaABaddBetaThis(1, y, GLOBAL::T, exxix, GLOBAL::N, 0); /*YtVRinv = y.GetTranspose() * exxix; */
//        realdp *YtVRinvptr = YtVRinv.ObtainWritePartialData();
//
//        for (integer i = 0; i < p; i++)
//        {
//            YtVRinvptr[i + p * i] = -YtVRinvptr[i + p * i];
//            for (integer j = i + 1; j < p; j++)
//            {
//                YtVRinvptr[i + p * j] = -YtVRinvptr[j + p * i] - YtVRinvptr[i + p * j];
//                YtVRinvptr[j + p * i] = 0;
//            }
//        }
//        Vector exresult(exxix); exresult.AlphaABaddBetaThis(1, y, GLOBAL::N, YtVRinv, GLOBAL::N, 1); /*exresult = exxix + y * YtVRinv; */
//
//        if(IsIntrApproach)
//            ObtainIntr(y, exresult, result);
//        else
//        {
//            *result = exresult;
//            ExtrProjection(y, *result, result);
//        }
//
//        if (IsEtaXiSameDir && HasHHR)
//        {
//            Vector beta(3);
//            realdp *betaptr = beta.ObtainWriteEntireData();
//            realdp EtatoXi = std::sqrt(Metric(x, etax, etax)) / nxix;
//            betaptr[0] = std::sqrt(Metric(x, etax, etax) / Metric(y, *result, *result)) / EtatoXi;
//            betaptr[1] = Metric(x, etax, etax);
//            betaptr[2] = Metric(y, *result, *result) * EtatoXi * EtatoXi;
//            etax.AddToFields("beta", beta);
//
//            if (HasHHR)
//            {
//                etax.AddToFields("betaTReta", (*result) * (betaptr[0] * EtatoXi));
//            }
//        }
//        return *result;
	};

	Vector &Grassmann::VecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
	{
        Vector Vt = y.Field("_Vt"), S = y.Field("_S");
        Vector Vtmp(Vt); S.DiagTimesM(Vtmp); Vtmp = Vt.GetTranspose() * Vtmp;
        if(IsIntrApproach)
        {
            Vector exxiy(EMPTYEXTR);
            ObtainExtr(y, xiy, &exxiy);
            Vector Vtmp2 = exxiy / Vtmp;
            Vector exresult = Vtmp2 - x * (x.GetTranspose() * Vtmp2);
            ObtainIntr(x, exresult, result);
            return *result;
        }
        
        Vector Vtmp2 = xiy / Vtmp;
        *result = Vtmp2 - x * (x.GetTranspose() * Vtmp2);
        
        return *result;
        
//        /* Below is the adjoint operator for the vector transport by the qf retraction */
//        Vector exxiy(EMPTYEXTR);
//        if(IsIntrApproach)
//            ObtainExtr(y, xiy, &exxiy);
//        else
//            exxiy = xiy;
//
//        Vector ytxiy(p, p); ytxiy.AlphaABaddBetaThis(1, y, GLOBAL::T, exxiy, GLOBAL::N, 0); /* ytxiy = y.GetTranspose() * exxiy;*/
//        realdp *ytxiyptr = ytxiy.ObtainWritePartialData();
//        for (integer i = 0; i < p; i++)
//        {
//            for (integer j = i; j < p; j++)
//            {
//                ytxiyptr[i + j * p] = -ytxiyptr[i + j * p];
//            }
//        }
//
//        Vector exresult = exxiy; exresult.AlphaABaddBetaThis(1, y, GLOBAL::N, ytxiy, GLOBAL::N, 1); /*exresult = y * ytxiy + exxiy;*/
//
//        realdp *exresultptr = exresult.ObtainWritePartialData();
//        Vector HHR = y.Field("_HHR");
//        const realdp *HHRptr = HHR.ObtainReadData();
//        for (integer i = 0; i < p; i++)
//        {
//            if(HHRptr[i + i * n] < 0)
//                scal_(&n, &GLOBAL::DNONE, exresultptr + i * n, &GLOBAL::IONE);
//        }
//
//        trsm_(GLOBAL::R, GLOBAL::U, GLOBAL::T, GLOBAL::N, &n, &p, &GLOBAL::DONE, const_cast<realdp *> (HHRptr), &n, exresultptr, &n);
//        ExtrProjection(x, exresult, &exresult);
//
//        if (IsIntrApproach)
//        {
//            return ObtainIntr(x, exresult, result);
//        }
//        *result = exresult;
//        return *result;
	};

    Vector &Grassmann::InverseVecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
    {
        Vector Vt = y.Field("_Vt"), S = y.Field("_S");
        Vector Vtmp(Vt); S.DiagTimesM(Vtmp); Vtmp = Vt.GetTranspose() * Vtmp;
        Vector Vtmp2 = xix * Vtmp;
        Vector YtX = y.GetTranspose() * x;
        *result = Vtmp2 - x * (YtX % (y.GetTranspose() * Vtmp2));
        return *result;
    };

    Vector &Grassmann::InverseVecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result, bool IsEtaXiSameDir) const
    {
        Vector Vt = y.Field("_Vt"), S = y.Field("_S");
        Vector Vtmp(Vt); S.DiagTimesM(Vtmp); Vtmp = Vt.GetTranspose() * Vtmp;
        Vector P = xiy * Vtmp;
        Vector A = (-1) * (x.GetTranspose() * y) % (x.GetTranspose() * P);
        *result = y * A + P;
        return *result;
    };

	realdp Grassmann::Beta(const Variable &x, const Vector &etax) const
	{
        if (!HasHHR)
            return 1;
        
        assert(etax.FieldsExist("beta"));

        /*If the beta has been computed in differentiated retraction, then obtain it.
        Beta should be almost always computed before.*/
        Vector beta = etax.Field("beta");
        const realdp *betav = beta.ObtainReadData();
        return betav[0];
	};

	Vector &Grassmann::EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const
	{
		if (prob->GetUseHess())
		{
            /*The copy on write is necessary. The reason is that the egf may be from a component in a product of elements.
            Therefore, if CopyOnWrite is not used, then the attached data in x and the product of elements share the same
            memory. This may cause an issue: if the product of elements are released before the attached data in x, then
            release the attached data in x would attempt to delete memory that has been released. This is an error!*/
            Vector EGrad(egf);
            EGrad.CopyOnWrite();
            x.AddToFields("EGrad", EGrad);
		}
		return ExtrProjection(x, egf, result);
	};

	Vector &Grassmann::EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const
	{
        Vector EGrad = x.Field("EGrad");
        Vector tmp(p, p); tmp.AlphaABaddBetaThis(1, x, GLOBAL::T, EGrad, GLOBAL::N, 0); /* tmp = x.GetTranspose() * EGrad; */
        *result = exix; result->AlphaABaddBetaThis(-1, etax, GLOBAL::N, tmp, GLOBAL::N, 1);
        return ExtrProjection(x, *result, result);
	};

	Vector &Grassmann::ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const
	{
        if(!x.FieldsExist("_HHR"))
        {
            x.HHRDecom();
        }
        Vector tmp = etax.HHRMtp(x.Field("_HHR"), x.Field("_tau"), GLOBAL::T, GLOBAL::L);
        
        const realdp *tmpptr = tmp.ObtainReadData();
        realdp *resultptr = result->ObtainWriteEntireData();
        
        for (integer i = 0; i < p; i++)
        {
            integer nmp = n - p;
            copy_(&nmp, const_cast<realdp *>(tmpptr + p + n* i), &GLOBAL::IONE, resultptr + nmp * i, &GLOBAL::IONE);
        }
        
        return *result;
	};

	Vector &Grassmann::ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const
	{
        if(!x.FieldsExist("_HHR"))
        {
            x.HHRDecom();
        }
        realdp *resultptr = result->ObtainWriteEntireData();
        const realdp *intretaxptr = intretax.ObtainReadData();

        for (integer i = 0; i < p; i++)
        {
            for(integer j = 0; j < p; j++)
            {
                resultptr[j + i * n] = 0;
            }
            integer nmp = n - p;
            copy_(&nmp, const_cast<realdp *> (intretaxptr)+nmp * i, &GLOBAL::IONE, resultptr + p + n* i, &GLOBAL::IONE);
        }

        (*result) = result->HHRMtp(x.Field("_HHR"), x.Field("_tau"), GLOBAL::N, GLOBAL::L);
        
        return *result;
	};

    Vector &Grassmann::ObtainNorVerIntr(const Variable &x, const Vector &etax, Vector *result) const
    {
        result->AlphaABaddBetaThis(1, x, GLOBAL::T, etax, GLOBAL::N, 0); /*xTetax  = x.GetTranspose() * etax; */
        
        return *result;
    };

    Vector &Grassmann::ObtainNorVerExtr(const Variable &x, const Vector &intretax, Vector *result) const
    {
        result->AlphaABaddBetaThis(1, x, GLOBAL::N, intretax, GLOBAL::N, 0);
        return *result;
    };

}; /*end of ROPTLIB namespace*/
