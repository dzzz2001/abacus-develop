/*
This file defines the class for the Poincare ball.

Manifold --> PoincareBall

---- WH
*/

#ifndef POINCAREBALL_H
#define POINCAREBALL_H

#include "Manifolds/Manifold.h"
#include "Others/def.h"
#include <cmath>

/*Define the namespace*/
namespace ROPTLIB {

	enum PoincareMetric { POINCARE_EUCLIDEAN, POINCARE_METRIC, POINCAREMETRICLENGTH };

	enum PoincareRetraction { POINCARE_FIRSEORDER, POINCARE_EXP,POINCARERETRACTIONLENGTH };

	enum PoincareVectorTransport { POINCARE_PARATRAN, POINCARE_VTPARA, POINCAREVECTORTRANSPORTLENGTH };

	class PoincareBall : public Manifold {
	public:
		/*Construct the PoincareBall manifold*/
		PoincareBall(integer inn);

		/*Delete EMPTYINTR and EMPTYEXTR*/
		virtual ~PoincareBall(void);

		/* choose Poincare metric, first order retraction, parallelization and intrinsic representation*/
		virtual void ChooseParamsSet1(void);

		/* choose Poincare metric, EXP retraction, parallelization and intrinsic representation*/
		virtual void ChooseParamsSet2(void);

		/* choose Poincare metric, first order retraction, parallel transport and extrinsic representation*/
		virtual void ChooseParamsSet3(void);

		/* choose Poincare metric, EXP retraction, parallel transport and extrinsic representation*/
		virtual void ChooseParamsSet4(void);

		/*Randomly generate a point on the Poincare ball manifold*/
		virtual Variable RandominManifold(void) const;

		/*Check whether all the parameters are legal or not.*/
		virtual void CheckParams(void) const;

		/* metrics */
		virtual realdp Metric(const Variable &x, const Vector &etax, const Vector &xix) const;

		/*The first order approximation of the exponential mapping, i.e., result = x + etax */
		virtual Variable &Retraction(const Variable &x, const Vector &etax, Variable *result) const;

		/*compute the distance between two points on the manifold.
		Default: the distance Poincare metric: 
		d(x1,x2) = arcosh( 1 + 2 * \frac{\| x1 - x2 \|^2}{(1 - \| u \|^2) * ( 1 - \| v \|^2)} )*/
		virtual realdp Dist(const Variable &x1, const Variable &x2) const;

		///*Compute the partial derivative of Dist for both parameters. */
		//virtual void DistPartialDeri(const Variable &x1, const Variable &x2, Vector *LEucGrad, Vector *REucGrad) const;

		/* Compute the intrinsic representation of a tagnent vector etax */
		virtual Vector &ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the extrinsic representation of a tagnent vector intretax. Inverse operation of the function ObtainIntr */
		virtual Vector &ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const;

		/*gf = \frac{(1 - \| \theta_t \|^2)^2}{4} * egf */
		virtual Vector &EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const;

		/*Computes the vector transport, i.e., result = \mathcal{T}_etax (xix)*/
		virtual Vector &VectorTransport(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const;

		/*Not done yet. Temporarily use: xix <-- exix*/
		virtual Vector &EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const;

	protected:
		integer n; /*The size of the space*/
		PoincareMetric metric; /*Riemannian metric*/
		PoincareRetraction retr;
		PoincareVectorTransport VecTran;

		/*The first order approximation of the exponential mapping, i.e., result = x + etax */
		virtual Variable &FirOrdRetraction(const Variable &x, const Vector &etax, Variable *result) const;

		Variable &MobiusAdd(const Variable &x, const Variable &y, Variable * result) const;

		Variable &ExpRetraction(const Variable &x, const Vector &etax, Variable *result) const;

		Vector &gyr(const Variable &u, const Variable &v, const Vector &w, Vector *result) const;

		virtual Vector &ParallelTranslation(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const;

		virtual Vector &VectorTransportParallelization(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result) const;

		/*Compute the intrinsic representation of a tagnent vector etax */
		virtual Vector &ObtainIntrPoincare(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the extrinsic representation of a tagnent vector intretax. Inverse operation of the function ObtainIntrAF */
		virtual Vector &ObtainExtrPoincare(const Variable &x, const Vector &intretax, Vector *result) const;

		//*gf = \frac{(1 - \| \theta_t \|^2)^2}{4} * egf */
		virtual Vector &EucGradToGradPoincare(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const;

		/*compute the distance between two points on the manifold.
		Default: the distance under Poincare metric: 
		d(x1,x2) = arcosh( 1 + 2 * \frac{\| x1 - x2 \|^2}{(1 - \| u \|^2) * ( 1 - \| v \|^2)} )*/
		virtual realdp DistPoincare(const Variable &x1, const Variable &x2) const;

		/*Compute the intrinsic representation of a tagnent vector etax*/
		virtual Vector &ObtainIntrEuc(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the extrinsic representation of a tagnent vector intretax. Inverse operation of the function ObtainIntrEuc */
		virtual Vector &ObtainExtrEuc(const Variable &x, const Vector &intretax, Vector *result) const;

		/*identity: gf = egf */
		virtual Vector &EucGradToGradEuc(const Variable &x, const Vector &egf, const Problem *prob, Vector *result) const;

		/*compute the distance between two points on the manifold.
		Default: the distance under Poincare metric: ||log(x1^{-1/2) x2 x1^{-1/2}||_F*/
		virtual realdp DistEuc(const Variable &x1, const Variable &x2) const;

	};
};/*end of ROPTLIB namespace*/

#endif /* end of POINCAREBALL_H */
