
#include "Manifolds/FixedRankE3F.h"

/*Define the namespace*/
namespace ROPTLIB{

	FixedRankE3F::FixedRankE3F(integer inm, integer inn, integer inr) : MultiManifolds(3,
		new Grassmann(inm, inr), static_cast<integer> (1), new Euclidean(inr, inr), static_cast<integer> (1), new Grassmann(inn, inr), static_cast<integer> (1))
	{
		m = inm;
		n = inn;
		r = inr;
		name.assign("FixedRankE3F");
        IsIntrApproach = true;
        
        Vector F1(m, r), F2(r, r), F3(n, r);
        Vector Prod(3, &F1, 1, &F2, 1, &F3, 1);
        EMPTYEXTR = Prod;
        Vector F4(m - r, r), F5(r, r), F6(n - r, r);
        Vector Prod2(3, &F4, 1, &F5, 1, &F6, 1);
        EMPTYINTR = Prod2;
        Vector F7(r, r), F8(0), F9(r, r);
        Vector Prod3(3, &F7, 1, &F8, 1, &F9, 1);
        EMPTYNORINTR = Prod3;
	};

	FixedRankE3F::~FixedRankE3F()
	{
        for (integer i = 0; i < numoftypes; i++)
        {
            delete manifolds[i];
        }
	};

	//realdp FixedRankE3F::Metric(Variable *x, Vector *etax, Vector *xix) const
	//{
	//	Vector *exetax = EMPTYEXTR->ConstructEmpty();
	//	Vector *exxix = EMPTYEXTR->ConstructEmpty();

	//	ObtainExtr(x, etax, exetax);
	//	ObtainExtr(x, xix, exxix);

	//	realdp result = ExtrMetric(x, exetax, exxix);

	//	delete exetax;
	//	delete exxix;
	//	return result;
	//};

	realdp FixedRankE3F::ExtrMetric(const Variable &x, const Vector &etax, const Vector &xix) const
	{ /* trace(D^T \dot{U}_1^T \dot{U}_2 D) + \trace(\dot{D}_1^T \dot{D}_2) + \trace(D \dot{V}_1^T \dot{V}_2 D^T) */
        Vector dU1 = etax.GetElement(0), dD1 = etax.GetElement(1), dV1 = etax.GetElement(2);
        Vector dU2 = xix.GetElement(0), dD2 = xix.GetElement(1), dV2 = xix.GetElement(2);
        Vector D = x.GetElement(1);
        
        Vector cdU1 = dU1 * D, cdU2 = dU2 * D;
        Vector cdV1 = dV1 * D.GetTranspose(), cdV2 = dV2 * D.GetTranspose();
        
        return cdU1.DotProduct(cdU2) + dD1.DotProduct(dD2) + cdV1.DotProduct(cdV2);
	};

	Vector &FixedRankE3F::ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const
	{
        result->NewMemoryOnWrite();
        manifolds[0]->ObtainIntr(x.GetElement(0), etax.GetElement(0), &result->GetElement(0));
        manifolds[1]->ObtainIntr(x.GetElement(1), etax.GetElement(1), &result->GetElement(1));
        manifolds[2]->ObtainIntr(x.GetElement(2), etax.GetElement(2), &result->GetElement(2));
        
        Vector D = x.GetElement(1);
        if(m - r > 0)
        {
            Vector tmp = result->GetElement(0) * D;
            result->GetElement(0) = tmp;
        }
        if(n - r > 0)
        {
            Vector tmp = result->GetElement(2) * D.GetTranspose();
            result->GetElement(2) = tmp;
        }
        return *result;
	};

	Vector &FixedRankE3F::ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const
	{
        result->NewMemoryOnWrite();
        
        Vector D = x.GetElement(1);
        if(m - r > 0)
            manifolds[0]->ObtainExtr(x.GetElement(0), intretax.GetElement(0) / D, &result->GetElement(0));
        
        result->GetElement(1) = intretax.GetElement(1);
        
        if(n - r > 0)
            manifolds[2]->ObtainExtr(x.GetElement(2), intretax.GetElement(2) / D.GetTranspose(), &result->GetElement(2));
        return *result;
	};

	Variable &FixedRankE3F::Retraction(const Variable &x, const Vector &etax, Variable *result) const
	{
		if (IsIntrApproach)
		{
            Vector exetax(EMPTYEXTR);
			ObtainExtr(x, etax, &exetax);
			SetIsIntrApproach(false);
			ProductManifold::Retraction(x, exetax, result);
			SetIsIntrApproach(true);
		}
		else
		{
			ProductManifold::Retraction(x, etax, result);
		}
        return *result;
	};

	Vector &FixedRankE3F::VecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const
	{
        Vector exetax(EMPTYEXTR), exxiy(EMPTYEXTR);
        ObtainExtr(x, etax, &exetax);
        ObtainExtr(y, xiy, &exxiy);
        for (integer i = 0; i < numoftypes; i++)
        {
            manifolds[i]->SetIsIntrApproach(false);
        }
        Vector exxiyU = exxiy.GetElement(0), exxiyD = exxiy.GetElement(1), exxiyV = exxiy.GetElement(2);
        Vector Uy = y.GetElement(0), Dy = y.GetElement(1), Vy = y.GetElement(2);
        
        Vector Utmp = exxiyU * Dy;
        exxiyU.AlphaABaddBetaThis(1, Utmp, GLOBAL::N, Dy, GLOBAL::T, 0); /* exxiyU = Utmp * Dy.GetTranspose(); */
        Utmp.AlphaABaddBetaThis(1, Uy, GLOBAL::N, exxiyD, GLOBAL::N, 0); /* Utmp = Uy * exxiyD; */
        exxiyU.AlphaABaddBetaThis(1, Utmp, GLOBAL::N, Dy, GLOBAL::T, 1); /* exxiyU = exxiyU + Utmp * Dy.GetTranspose()*/
        exxiy.GetElement(0) = exxiyU;
        
        Vector Vtmp(n, r); Vtmp.AlphaABaddBetaThis(1, exxiyV, GLOBAL::N, Dy, GLOBAL::T, 0); /* Vtmp = exxiyV * Dy.GetTranspose(); */
        exxiyV.AlphaABaddBetaThis(1, Vtmp, GLOBAL::N, Dy, GLOBAL::N, 0);
        Vtmp.AlphaABaddBetaThis(1, Vy, GLOBAL::N, exxiyD, GLOBAL::T, 0);
        exxiyV.AlphaABaddBetaThis(1, Vtmp, GLOBAL::N, Dy, GLOBAL::N, 1);
        exxiy.GetElement(2) = exxiyV;
        
        exxiyU.Delete(); exxiyD.Delete(); exxiyV.Delete();
        
        manifolds[0]->ExtrProjection(y.GetElement(0), exxiy.GetElement(0), &exxiy.GetElement(0));
        manifolds[2]->ExtrProjection(y.GetElement(2), exxiy.GetElement(2), &exxiy.GetElement(2));
        
        Vector exresult(EMPTYEXTR);
        exresult.NewMemoryOnWrite();
        ProductManifold::VecTranDiffRetAdjoint(x, exetax, y, exxiy, &exresult);
		ExtrProjectionStiePerp(x.GetElement(0), exresult.GetElement(0), &exresult.GetElement(0));
        ExtrProjectionStiePerp(x.GetElement(2), exresult.GetElement(2), &exresult.GetElement(2));

        Vector D = x.GetElement(1), Dt = D.GetTranspose();
        Vector tmp1 = exresult.GetElement(0) / Dt / D, tmp2 = exresult.GetElement(2) / D / Dt;
        exresult.GetElement(0) = tmp1;
        exresult.GetElement(2) = tmp2;
        ObtainIntr(x, exresult, result);
		ObtainIntr(x, exresult, result);
		for (integer i = 0; i < numoftypes; i++)
		{
			manifolds[i]->SetIsIntrApproach(true);
		}
        return *result;
	};

    Vector &FixedRankE3F::ExtrProjectionStiePerp(const Variable &x, const Vector &v, Vector *result) const
    {
        integer p = x.Getcol();
        Vector UtV(p, p); UtV.AlphaABaddBetaThis(1, x, GLOBAL::T, v, GLOBAL::N, 0);
        *result = v;
        result->AlphaABaddBetaThis(-1, x, GLOBAL::N, UtV, GLOBAL::N, 1);
        return *result;
    };

	Vector &FixedRankE3F::VecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result, bool IsEtaXiSameDir) const
	{
        realdp nxix = std::sqrt(Metric(x, xix, xix));
        
        Vector exetax(EMPTYEXTR), exxix(EMPTYEXTR), exresult(EMPTYEXTR);
        ObtainExtr(x, etax, &exetax);
        ObtainExtr(x, xix, &exxix);
		for (integer i = 0; i < numoftypes; i++)
		{
			manifolds[i]->SetIsIntrApproach(false);
		}
        exresult.NewMemoryOnWrite();
        manifolds[0]->VecTranDiffRet(x.GetElement(0), exetax.GetElement(0), y.GetElement(0), exxix.GetElement(0), &exresult.GetElement(0), IsEtaXiSameDir);
        manifolds[1]->VecTranDiffRet(x.GetElement(1), exetax.GetElement(1), y.GetElement(1), exxix.GetElement(1), &exresult.GetElement(1), IsEtaXiSameDir);
        manifolds[2]->VecTranDiffRet(x.GetElement(2), exetax.GetElement(2), y.GetElement(2), exxix.GetElement(2), &exresult.GetElement(2), IsEtaXiSameDir);
        
		ObtainIntr(y, exresult, result);
		for (integer i = 0; i < numoftypes; i++)
		{
			manifolds[i]->SetIsIntrApproach(true);
		}
        
        if (IsEtaXiSameDir && HasHHR)
        {
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
	};

	Vector &FixedRankE3F::ExtrProjection(const Variable &x, const Vector &etax, Vector *result) const
	{
//        Vector inetax(EMPTYINTR);
//        ObtainIntr(x, etax, &inetax);
//        ObtainExtr(x, inetax, result);
        MultiManifolds::ExtrProjection(x, etax, result);
        return *result;
	};

	Vector &FixedRankE3F::EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const
	{
        /*The Euclidean is a m by n dense or sparse matrix stored in a field with name "DenseMatrix" or "_SparseMatrix"
        When converting the Euclidean format to Extrinsic format, the main data in egf has to be updated. Therefore,
        egf is casted to be a non-const variable*/
        EucRepToExtr(x, const_cast<Vector *> (&egf));
        if(prob->GetUseHess())
        {
            /*The copy on write is necessary. The reason is that the egf may be from a component in a product of elements.
            Therefore, if CopyOnWrite is not used, then the attached data in x and the product of elements share the same
            memory. This may cause an issue: if the product of elements are released before the attached data in x, then
            release the attached data in x would attempt to delete memory that has been released. This is an error!*/
            Vector Sharedegf(egf);
            Sharedegf.CopyOnWrite();
            x.AddToFields("EGrad", Sharedegf);
        }
        *result = egf;
		return *result;
	};

	Vector &FixedRankE3F::EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const
	{
        EucRepToExtr(x, const_cast<Vector *> (&exix));
        *result = exix;
        
        Vector segf = x.Field("EGrad");
        Vector dU = etax.GetElement(0), dV = etax.GetElement(2);
        /*MdV = M * dV, MtdU = M^T * dU */
        Vector MdV(m, r), MtdU(n, r);
        if(segf.FieldsExist("DenseMatrix"))
        {
            Vector M = segf.Field("DenseMatrix");
            MdV.AlphaABaddBetaThis(1, M, GLOBAL::N, dV, GLOBAL::N, 0);
            MtdU.AlphaABaddBetaThis(1, M, GLOBAL::T, dU, GLOBAL::N, 0);
        } else
        if(segf.FieldsExist("_SparseMatrix"))
        {
            SparseMatrix *sM = segf.GetSparseMatrixinFields();
            MdV = (*sM) * dV;
            MtdU = (dU.GetTranspose() * (*sM)).GetTranspose();
        } else
        {
            assert(false);
        }
        
		MTdUMdVtoExtr(x, MtdU, MdV, result);
        return *result;
	};

	Vector &FixedRankE3F::MTdUMdVtoExtr(const Variable &x, Vector &MTdU, Vector &MdV, Vector* xix) const
	{
        Vector U = x.GetElement(0), D = x.GetElement(1), V = x.GetElement(2);
        
        /*Compute dU_R = (I - U U^T) MdV D^{-1} */
        Vector UtMdV(r, r); UtMdV.AlphaABaddBetaThis(1, U, GLOBAL::T, MdV, GLOBAL::N, 0);
        MdV.AlphaABaddBetaThis(-1, U, GLOBAL::N, UtMdV, GLOBAL::N, 1);
        MdV = MdV / D;
        
        /*dU_xix = dU_xix + dU_R*/
        xix->CopyOnWrite();
//        Vector dU_xix = xix->GetElement(0), dV_xix = xix->GetElement(2);
        xix->GetElement(0).AlphaXaddThis(1, MdV);
        
        /*Compute dV_R = (I - V V^T) MTdU D^{-T} */
        Vector VtMtdU(r, r); VtMtdU.AlphaABaddBetaThis(1, V, GLOBAL::T, MTdU, GLOBAL::N, 0);
        MTdU.AlphaABaddBetaThis(-1, V, GLOBAL::N, VtMtdU, GLOBAL::N, 1);
        MTdU = MTdU / D.GetTranspose();
        
        /*dV_xix = dV_xix + dV_R*/
        xix->GetElement(2).AlphaXaddThis(1, MTdU);
        return *xix;
	};

	Vector &FixedRankE3F::EucRepToExtr(const Variable &x, Vector *result) const
	{
        Vector M(m, n);
        const SparseMatrix *sM = nullptr;
        /*Compute MV = M * V or sM * V and MTV = M^T * U or sM^T * U */
        Vector MV(m, r), MTU(n, r);
        if(result->FieldsExist("DenseMatrix"))
        {
            M = result->Field("DenseMatrix");
            MV.AlphaABaddBetaThis(1, M, GLOBAL::N, x.GetElement(2), GLOBAL::N, 0);
            MTU.AlphaABaddBetaThis(1, M, GLOBAL::T, x.GetElement(0), GLOBAL::N, 0);
        } else
        if(result->FieldsExist("_SparseMatrix"))
        {
            sM = result->GetSparseMatrixinFields();
            MTU.AlphaABaddBetaThis(1, *sM, GLOBAL::T, x.GetElement(0), GLOBAL::N, 0);
            MV.AlphaABaddBetaThis(1, *sM, GLOBAL::N, x.GetElement(2), GLOBAL::N, 0);
        } else
        {
            assert(false);
        }
		MTUMVtoExtr(x, MTU, MV, result);
        return *result;
	};

	Vector &FixedRankE3F::MTUMVtoExtr(const Variable &x, Vector &MtU, Vector &MV, Vector *result) const
	{
        Vector U = x.GetElement(0), D = x.GetElement(1), V = x.GetElement(2);
        result->CopyOnWrite();
//        result->NewMemoryOnWrite();
        result->GetElement(1).AlphaABaddBetaThis(1, U, GLOBAL::T, MV, GLOBAL::N, 0);
        
        result->GetElement(0) = MV;
        result->GetElement(2) = MtU;
        /*compute MV - U dotD*/
        result->GetElement(0).AlphaABaddBetaThis(-1, U, GLOBAL::N, result->GetElement(1), GLOBAL::N, 1);
        /*compute MTU - V dotD^T*/
        result->GetElement(2).AlphaABaddBetaThis(-1, V, GLOBAL::N, result->GetElement(1), GLOBAL::T, 1);
        
        /* Compute dotU = (MV - U dotD) D^{-1} */
        Vector tmp1 = result->GetElement(0) / D;
        result->GetElement(0) = tmp1;
        
        /*Compute dotV = (MTU - V dotD) D^{-T} */
        Vector tmp2 = result->GetElement(2) / D.GetTranspose();
        result->GetElement(2) = tmp2;
        
        return *result;
	};

	Vector &FixedRankE3F::ExtrToEucRep(const Variable &x, Vector *result) const
	{
        if(result->FieldsExist("DenseMatrix") || result->FieldsExist("_SparseMatrix"))
            return *result;
        
        Vector U = x.GetElement(0), D = x.GetElement(1), V = x.GetElement(2);
        Vector dotU = result->GetElement(0), dotD = result->GetElement(1), dotV = result->GetElement(2);
        
        Vector M(m, n);
        *result = U * dotD * V.GetTranspose() + dotU * D * V.GetTranspose() + U * D * dotV.GetTranspose();
		result->AddToFields("DenseMatrix", M);
        return *result;
	}
}; /*end of ROPTLIB namespace*/
