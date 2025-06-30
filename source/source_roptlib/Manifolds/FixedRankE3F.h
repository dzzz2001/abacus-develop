/*
This file defines the class for the fixed-rank manifold R_r^{m times n}, which is represented by Gr(r, m) times R^{r times r} times Gr(r, n).
Note that unlike the manifold defined in FixedRankE, m by n matrix is not stored explictly in this class.

Since we represented a m by n matrix by an element in Gr(r, m) times R^{r times r} times Gr(r, n), the Euclidean gradient can not be computed by
finite difference. Therefore, the numerical gradient and numerical Hessian functionalities are not supported when a problem is defined on this
manifold representation.

A tangent vector in T_x R_r^{m times n} can representated by etax = dot{U} D V^T + U dot{D} V^T + U D dot{V}^T, where dot{U}^T U = 0, dot{V}^T V = 0.
In the implementation, a tangent vector has 3 representations:
1), extrinsic representation: (dot{U}, dot{D}, dot{V})
2), intrinsic representation: a (mr + nr - r^2) vector
3), Euclidean representation: a m-by-n dense or sparse matrix.
Given a Euclidean gradient, which is represented in the third form,
The Euclidean representation is stored in a temporary data with key "DenseMatrix" or "_SparseMatrix" in a EMPTYEXTR-type tangent vector.
See FRankE3FMatCompletion.h and FRankE3FMatCompletion.cpp to see an example of problem defined on this manifold.
This class provides a function to convert the Euclidean representation into the extrinsic representation.

The used Riemannian metric is
g(etax, xix) = trace(etax^T xix)
= trace(D^T \dot{U}_1^T \dot{U}_2 D) + \trace(\dot{D}_1^T \dot{D}_2) + \trace(D \dot{V}_1^T \dot{V}_2 D^T),
where etax = dot{U}_1 D V^T + U dot{D}_1 V^T + U D dot{V}_1^T and xix = dot{U}_2 D V^T + U dot{D}_2 V^T + U D dot{V}_2^T.

 Manifold --> MultiManifolds --> FixedRankE3F

---- WH
*/

#ifndef FIXEDRANKE3F_H
#define FIXEDRANKE3F_H

#include "Manifolds/Euclidean.h"
#include "Manifolds/MultiManifolds.h"
#include "Manifolds/Grassmann.h"
#include "Manifolds/Euclidean.h"

/*Define the namespace*/
namespace ROPTLIB{

	class FixedRankE3F : public MultiManifolds{
	public:
		/*Construct the fixed rank manifold of m by n matrices with rank r.
		It is represented by Gr(r, m) times R^{r times r} times Gr(r, n), i.e.,
		X = U D V^T. U \in Gr(r, m), D \in R^{r times r} and V \in Gr(r, n).
		Note that D is not necessary a diagonal matrix.*/
		FixedRankE3F(integer m, integer n, integer r);

		/*Delete the manifold by deleting each component.*/
		~FixedRankE3F(void);

		/*Riemannian metric*/
		//virtual realdp Metric(Variable *x, Vector *etax, Vector *xix) const;
		virtual realdp ExtrMetric(const Variable &x, const Vector &etax, const Vector &xix) const;

		/*Tangent vector etax = \dot(U) D V^T + U \dot{D} V^T + U D \dot{V}^T. Let \dot{U} = U_perp K_U and \dot{V} = V_perp K_V
		It follows that etax = U_perp K_U D V^T + U D K_V^T V_perp^T + U \dot{D} V^T
		The intrinsic representation would be given by vectorizing (A, B, C) := (K_U D, D K_V^T, \dot{D}) .*/
        virtual Vector &ObtainIntr(const Variable &x, const Vector &etax, Vector *result) const;

		/*Compute the extrinsic approach by given (A, B, C) := (K_U D, D K_V^T, \dot{D}).
		\dot{U} = U_\perp A D^{-1}, \dot{D} = C, \dot{V} = V_\perp B^T D^{-T}.*/
        virtual Vector &ObtainExtr(const Variable &x, const Vector &intretax, Vector *result) const;

		/*Perform the default retraction of each manifold component*/
        virtual Variable &Retraction(const Variable &x, const Vector &etax, Variable *result) const;

		/*Compute the tangent vector result satisfying
		g_y(\mathcal{T}_{R_etax}(xix), xiy) = g_x(xix, result) for all xix \in T_x M,
		where y = R_x(etax), xiy \in T_y M.*/
        virtual Vector &VecTranDiffRetAdjoint(const Variable &x, const Vector &etax, const Variable &y, const Vector &xiy, Vector *result) const;
        
        /*Used in "VecTranDiffRetAdjointCan".*/
        virtual Vector &ExtrProjectionStiePerp(const Variable &x, const Vector &v, Vector *result) const;
        
		/*Perform the vector transport by differentiated retraction of each individual manifold.*/
        virtual Vector &VecTranDiffRet(const Variable &x, const Vector &etax, const Variable &y, const Vector &xix, Vector *result, bool IsEtaXiSameDir = false) const;

		/*etax is in the ambient space R^{m \times r} \times R^{r \times r} \times R^{n \times r}. This function projects etax onto the
		tangent space of x, i.e., result = P_{T_x M} v, where P is based on the selected Riemannian metric*/
        virtual Vector &ExtrProjection(const Variable &x, const Vector &etax, Vector *result) const;

		/*the Riemannian gradient is obtained by projecting the Euclidean onto the tangent space of x.*/
        virtual Vector &EucGradToGrad(const Variable &x, const Vector &egf, const Problem *prob, Vector *result)  const;

		/*The Riemannian action of the Hessian is obtained by Hess f(x)[etax] = P_x(D grad f(x) [etax]).*/
        virtual Vector &EucHvToHv(const Variable &x, const Vector &etax, const Vector &exix, const Problem *prob, Vector *result) const;

		/*Convert the Euclidean representation R^{m \times n} of a tangent vector to the Extrinsic representation.
		The Euclidean representation is attached in the temporary data of "result" with key "EucRep". */
		virtual Vector &EucRepToExtr(const Variable &x, Vector *result) const;

		/*Convert the Extrinsic representation of a tangent vector to the Euclidean representation.
		The Euclidean representation is attached in the temporary data of "result" with key "EucRep". */
		virtual Vector &ExtrToEucRep(const Variable &x, Vector *result) const;

		/*M is in Euclidean representation. This function computes extrinsic representation when
		M^T U and M V are computed*/
		Vector &MTUMVtoExtr(const Variable &x, Vector &MtU, Vector &MV, Vector *result) const;

		/*This is used in EucHvToHv */
		Vector &MTdUMdVtoExtr(const Variable &x, Vector &MTdU, Vector &MdV, Vector* xix) const;

	protected:
		mutable integer m; /*the number of row*/
		mutable integer n; /*the number of column*/
		mutable integer r; /*the rank of the matrix*/
	};
}; /*end of ROPTLIB namespace*/
#endif // end of LOWRANK_H
