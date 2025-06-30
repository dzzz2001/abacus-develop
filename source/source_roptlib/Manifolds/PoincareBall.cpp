#include "Manifolds/PoincareBall.h"

/*Define the namespace*/
namespace ROPTLIB {

	PoincareBall::PoincareBall(integer inn)
	{
		n = inn;
		IsIntrApproach = true;
		HasHHR = false;
		name.assign("PoincareBall");
		IntrinsicDim = n;
		ExtrinsicDim = n;
		metric = POINCARE_METRIC;
		retr = POINCARE_FIRSEORDER;
		VecTran = POINCARE_VTPARA;
		EMPTYEXTR = Vector(n);
		EMPTYINTR = Vector(IntrinsicDim);
	};

	PoincareBall::~PoincareBall(void)
	{
	};

	void PoincareBall::ChooseParamsSet1(void)
	{
		IsIntrApproach = true;
		metric = POINCARE_METRIC;
		retr = POINCARE_FIRSEORDER;
		VecTran = POINCARE_VTPARA;
	};

	void PoincareBall::ChooseParamsSet2(void)
	{
		IsIntrApproach = true;
		metric = POINCARE_METRIC;
		retr = POINCARE_EXP;
		VecTran = POINCARE_VTPARA;
	};

	void PoincareBall::ChooseParamsSet3(void)
	{
		IsIntrApproach = false;
		metric = POINCARE_METRIC;
		retr = POINCARE_FIRSEORDER;
		VecTran = POINCARE_PARATRAN;
	};

	void PoincareBall::ChooseParamsSet4(void)
	{
		IsIntrApproach = false;
		metric = POINCARE_METRIC;
		retr = POINCARE_EXP;
		VecTran = POINCARE_PARATRAN;
	};

	realdp PoincareBall::Metric(const Variable &x, const Vector &etax, const Vector &xix) const
	{
		if (IsIntrApproach || metric == POINCARE_EUCLIDEAN)
			return etax.DotProduct(xix);

		if (metric == POINCARE_METRIC)
		{
			realdp lambda = 2 / (1 - x.DotProduct(x));
			return lambda * lambda * etax.DotProduct(xix);
		}

		printf("Warning: PoincareBall::metric has not been done!\n");
		return etax.DotProduct(xix);
	};

	Variable PoincareBall::RandominManifold(void) const
	{
		Vector tmp(n);
		tmp.RandUnform(-0.001, 0.001);
		//tmp.RandUnform(-1, 1);
		return tmp;
	};

	void PoincareBall::CheckParams(void) const
	{
		std::string PoincareMetricnames[POINCAREMETRICLENGTH] = { "EUCLIDEAN", "POINCARE" };
		std::string PoincareRetractionnames[POINCARERETRACTIONLENGTH] = { "POINCAREFIRSEORDER", "POINCAREEXP" };
		std::string PoincareVecTrannames[POINCAREVECTORTRANSPORTLENGTH] = { "POINCAREPARATRAN","POINCAREVTPARA" };
		Manifold::CheckParams();
		printf("%s PARAMETERS:\n", name.c_str());
		printf("n           :%15d,\t", n);
		printf("metric        :%15s,\t", PoincareMetricnames[metric].c_str());
		printf("Retraction    :%15s,\n", PoincareRetractionnames[retr].c_str());
		printf("VecTran       :%15s,\n", PoincareVecTrannames[VecTran].c_str());

	};


	Vector &PoincareBall::EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const
	{
		if (metric == POINCARE_METRIC)
			return EucGradToGradPoincare(x, egf, prob, result);

		if (metric == POINCARE_EUCLIDEAN)
			return EucGradToGradEuc(x, egf, prob, result);

		printf("Warning: the Riemannian metric is not valid! Use the Poincare metric instead!");
		return EucGradToGradPoincare(x, egf, prob, result);
	};

	Vector &PoincareBall::EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const
	{
		*result = exix;
		return *result;
	};

	Vector &PoincareBall::ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const
	{
		if (metric == POINCARE_METRIC)
			return ObtainIntrPoincare(x, etax, result);

		if (metric == POINCARE_EUCLIDEAN)
			return ObtainIntrEuc(x, etax, result);

		printf("Warning: the Riemannian metric is not valid! Use the Poincare metric instead!");
		return ObtainIntrPoincare(x, etax, result);
	};

	Vector &PoincareBall::ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const
	{
		if (metric == POINCARE_METRIC)
			return ObtainExtrPoincare(x, intretax, result);

		if (metric == POINCARE_EUCLIDEAN)
			return ObtainExtrEuc(x, intretax, result);

		printf("Warning: the Riemannian metric is not valid! Use the Poincare metric instead!");
		return ObtainExtrPoincare(x, intretax, result);
	};

	realdp PoincareBall::Dist(const Variable &x1, const Variable &x2) const
	{
		if (metric == POINCARE_METRIC)
			return DistPoincare(x1, x2);

		if (metric == POINCARE_EUCLIDEAN)
			return DistEuc(x1, x2);

		printf("Warning: the Riemannian metric is not valid! Use the Poincare metric instead!");
		return DistPoincare(x1, x2);
	};

	Variable &PoincareBall::Retraction(const Variable &x, const Vector &etax, Variable *result) const
	{
		if (retr == POINCARE_FIRSEORDER)
			return FirOrdRetraction(x, etax, result);

		if (retr == POINCARE_EXP)
			return ExpRetraction(x, etax, result);

		printf("Warning: the Riemannian retraction is not valid! Use the first order retraction instead!");
		return FirOrdRetraction(x, etax, result);
	};

	Variable &PoincareBall::FirOrdRetraction(const Variable &x, const Vector &etax, Variable *result) const
	{	/* result = x + etx  */

		const realdp *etaxptr = etax.ObtainReadData();
		bool flag = false;
		for (integer i = 0; i < etax.Getlength(); i++)
		{
			if (std::abs(etaxptr[i]) > 1e-8)
			{
				flag = true;
				break;
			}
		}
		if (!flag)
		{
			*result = x;
			return *result;
		}

		Vector exetax(EMPTYEXTR);
		Vector tmp(EMPTYEXTR);

		if (IsIntrApproach)
			ObtainExtr(x, etax, &exetax);
		else
			exetax = etax;

		tmp = exetax.AlphaXaddThis(1.0, x); // x + exetax

		realdp norm_tmp = tmp.DotProduct(tmp);
		realdp eps = 1e-5;

		if (norm_tmp >= 1)
			*result = tmp / std::sqrt(norm_tmp) - eps;
		else
			*result = tmp;
		
		return *result;
	};

	Variable &PoincareBall::MobiusAdd(const Variable &x, const Variable &y, Variable * result) const
	{
		realdp xy = x.DotProduct(y);
		realdp Normx = x.DotProduct(x);
		realdp Normy = y.DotProduct(y);
		realdp tmp = 1 + 2 * xy + Normx * Normy;
		result->SetToZeros();
		result->AlphaXaddThis((1 + 2 * xy + Normy)/tmp, x);
		result->AlphaXaddThis((1 - Normx)/tmp, y);
		return *result;
	};

	Variable &PoincareBall::ExpRetraction(const Variable &x, const Vector &etax, Variable *result) const
	{
		const realdp *etaxptr = etax.ObtainReadData();
		bool flag = false;
		for (integer i = 0; i < etax.Getlength(); i++)
		{
			if (std::abs(etaxptr[i]) > 1e-8)
			{
				flag = true;
				break;
			}
		}
		if (!flag)
		{
			*result = x;
			return *result;
		}

		Vector exetax(EMPTYEXTR);
		if (IsIntrApproach)
			ObtainExtr(x, etax, &exetax);
		else
			exetax = etax;

		Variable tmp(n);
		realdp Normetax = std::sqrt(exetax.DotProduct(exetax));
		tmp.SetToZeros();
		tmp.AlphaXaddThis(std::tanh(Normetax / (1 - x.DotProduct(x))) / Normetax, exetax);
		MobiusAdd(x, tmp, result);
		
		return *result;
	};

	Vector &PoincareBall::EucGradToGradPoincare(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const
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

		realdp lambda = (1 - x.DotProduct(x)) / 2;
		result->SetToZeros();
		result->AlphaXaddThis(lambda * lambda, egf);

		return *result;
	};

	Vector &PoincareBall::ObtainIntrPoincare(const Variable &x, const Vector &etax, Vector *result) const
	{
		realdp x_norm2 = x.DotProduct(x);
		realdp lambda = 2 / (1 - x_norm2);
		result->SetToZeros();
		result->AlphaXaddThis(lambda, etax);
		return *result;
	};

	Vector &PoincareBall::ObtainExtrPoincare(const Variable &x, const Vector &intretax, Vector *result) const
	{
		realdp x_norm2 = x.DotProduct(x);
		realdp invlambda = (1 - x_norm2) /2;
		result->SetToZeros();
		result->AlphaXaddThis(invlambda, intretax);
		return *result;
	};


	realdp PoincareBall::DistPoincare(const Variable &x1, const Variable &x2) const
	{/*d(x1,x2) = arcosh( 1 + 2 * \frac{\| x1 - x2 \|^2}{(1 - \| u \|^2) * ( 1 - \| v \|^2)} )*/
		Vector u_minus_v(n);
		u_minus_v = x1 - x2;
		realdp u_minus_v2 = u_minus_v.DotProduct(u_minus_v);
		realdp uu = x1.DotProduct(x1);
		realdp vv = x2.DotProduct(x2);
		realdp alpha = 1.0 - uu;
		realdp beta = 1.0 - vv;
		realdp gamma = 1.0 + 2.0 * u_minus_v2 / alpha / beta ;
		if (gamma < 1)
			gamma = 1; // for nemerical error
		
		return std::acosh(gamma);
	};



	Vector &PoincareBall::EucGradToGradEuc(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const
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
		*result = egf;
		return *result;
	};

	Vector &PoincareBall::VectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
	{
		if (VecTran == POINCARE_VTPARA)
			return VectorTransportParallelization(x, etax, y, xix, result);

		if (VecTran == POINCARE_PARATRAN)
			return ParallelTranslation(x, etax, y, xix, result);

		printf("Warning: the VectorTransport is not valid! Use vector transport by parallelization instead!");
		return VectorTransportParallelization(x, etax, y, xix, result);
	};

	Vector &PoincareBall::gyr(const Variable &u, const Variable &v, const Vector &w, Vector *result) const
	{
		realdp uw = u.DotProduct(w);
		realdp vw = v.DotProduct(w);
		realdp uv = u.DotProduct(v);
		realdp uu = u.DotProduct(u);
		realdp vv = v.DotProduct(v);
		realdp A = -uw * vv + vw + 2 * uv * vw;
		realdp B = -vw * uu - uw;
		realdp D = 1 + 2 * uv + uu * vv;
		result->SetToZeros();
		result->AlphaXaddThis(2 * A / D, u);
		result->AlphaXaddThis(2 * B / D, v);
		result->AlphaXaddThis(1, w);

		return *result;
	};

	Vector &PoincareBall::ParallelTranslation(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
	{
		Vector exetax(EMPTYEXTR), exxix(EMPTYEXTR);
		if (IsIntrApproach)
		{
			ObtainExtr(x, xix, &exxix);
		}
		else
		{
			exxix = xix;
		}

		realdp lambdax = 2 / (1 - x.DotProduct(x));
		realdp lambday = 2 / (1 - y.DotProduct(y));
		Variable tmp(n);
		tmp = x;
		tmp.ScalarTimesThis(-1);
		gyr(y, tmp, exxix, result);
		result->ScalarTimesThis(lambdax / lambday);
		return *result;
	};

	Vector &PoincareBall::VectorTransportParallelization(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const
	{
		if (IsIntrApproach)
		{
			*result = xix;
			return *result;
		}
		Vector tmp(EMPTYEXTR); 
		ObtainIntr(x, xix, &tmp);
		return ObtainExtr(y, tmp, result);
	};

	Vector &PoincareBall::ObtainIntrEuc(const Variable &x, const Vector &etax, Vector *result) const
	{
		*result = etax;
		return *result;
	};

	Vector &PoincareBall::ObtainExtrEuc(const Variable &x, const Vector &intretax, Vector *result) const
	{
		*result = intretax;
		return *result;
	};

	realdp PoincareBall::DistEuc(const Variable &x1, const Variable &x2) const
	{
		Vector u_minus_v = x1 - x2;
		return u_minus_v.DotProduct(u_minus_v);
	};
}
