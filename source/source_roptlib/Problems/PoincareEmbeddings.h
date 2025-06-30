#ifndef POINCAREEMBEDDINGS_H
#define POINCAREEMBEDDINGS_H



#include "Manifolds/PoincareBall.h"
#include "Problems/Problem.h"
#include "Manifolds/MultiManifolds.h"
#include "Others/def.h"
#include "Others/randgen.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include <algorithm>

/*Define the namespace*/
namespace ROPTLIB {
	class PoincareEmbeddings : public Problem {
	public:
		PoincareEmbeddings(Vector indata, integer inn, integer inXNum, integer innegs, realdp inSampleDampening = 0.75);

		virtual ~PoincareEmbeddings();

		virtual Vector &NegativeSampler() const;

		virtual realdp f(const Variable &x) const;
        
        virtual realdp Stof(const Variable &x, const Vector &batch_index) const;
        
		virtual Vector &EucGrad(const Variable &x, Vector *result) const;
        
        virtual Vector &EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const;

		virtual void DistPartialDeriPoincare(const Variable &x1, const Variable &x2, Vector *LEucGrad, Vector *REucGrad) const;

		virtual void dataToGraph(void);

		/*Create the Alias table*/
		virtual void createAliasTable(void);
		 
		/*Negative sampling in burnin phase*/
		virtual integer RandomNode(void) const;

		/*Update the information for each epoch*/
		virtual void UpdateEpoch(void) const;

		Vector data;
		//Vector NegSamp; // This parameter has been removed and transferred to Problem.h
		mutable integer NegSampleNum; 

		integer n;
		integer XNum;
		realdp SampleDampening;
		realdp NegMultiplier;
		mutable Vector NegSamp; // store negative samples

	protected:
		std::unordered_map<integer, std::unordered_set<integer>> dataGraph;
		Vector S; //store probability distribution
		Vector A; //Alias table
		integer SearchNum;
	};
}; /*end of ROPTLIB namespace*/

#endif // end of POINCAREEMBEDDINGS_H
