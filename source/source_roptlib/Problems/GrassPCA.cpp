
#include "Problems/GrassPCA.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    GrassPCA::GrassPCA(Vector inA, integer inN, integer inn, integer inp)
    {
        A = inA;
        N = inN;
        n = inn;
        p = inp;
        NumGradHess = false;
    };
    
    GrassPCA::~GrassPCA(void)
    {
    };
    
    realdp GrassPCA::f(const Variable &x) const
    {
        realdp result = 0;
        Vector UTx(p);
        for (integer i = 0; i < N; i++)
        {
            UTx.AlphaABaddBetaThis(1, x, GLOBAL::T, A.GetSubmatrix(0, n - 1, i, i), GLOBAL::N, 0); /* Ax = U^T * x_i; */
            result = result + UTx.DotProduct(UTx);
        }
        result = - result / N;
        return result;   //test problem
    };

    realdp GrassPCA::Stof(const Variable &x, const Vector &batch_index) const
    {
        realdp result = 0;
        Vector UTx(p);
        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            UTx.AlphaABaddBetaThis(1, x, GLOBAL::T, A.GetSubmatrix(0, n - 1, static_cast<integer> (batch_index[i]), static_cast<integer> (batch_index[i])), GLOBAL::N, 0); /* Ax = U^T * x_i; */
            //std::cout <<  << std::endl;
            result = result + UTx.DotProduct(UTx);
        }
        result = - result / batch_index.Getlength();
        return result;   //test problem
    };

    Vector &GrassPCA::EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const
    {
        Vector xTU(1,p);
        Vector xxTU(n,p);
        result->SetToZeros();
        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            xTU.AlphaABaddBetaThis(1, A.GetSubmatrix(0, n - 1, static_cast<integer> (batch_index[i]), static_cast<integer> (batch_index[i])), GLOBAL::T, x, GLOBAL::N, 0);
            xxTU.AlphaABaddBetaThis(1, A.GetSubmatrix(0, n - 1, static_cast<integer> (batch_index[i]), static_cast<integer> (batch_index[i])), GLOBAL::N, xTU, GLOBAL::N, 0);
            result->AlphaXaddThis(1, xxTU);
        }
        result->ScalarTimesThis(-2. / batch_index.Getlength());
        return *result;
    };

    Vector &GrassPCA::EucGrad(const Variable &x, Vector *result) const
    {
        Vector xTU(1,p);
        Vector xxTU(n,p);
        result->SetToZeros();
        //tmpAi.PrintSize();
        //result->PrintSize();
        for (integer i = 0; i < N; i++)
        {
            xTU.AlphaABaddBetaThis(1, A.GetSubmatrix(0, n - 1, i, i), GLOBAL::T, x, GLOBAL::N, 0); /* xTU = x_i^T * U; */
            xxTU.AlphaABaddBetaThis(1, A.GetSubmatrix(0, n - 1, i, i), GLOBAL::N, xTU, GLOBAL::N, 0);
            result->AlphaXaddThis(1, xxTU);
        }
        result->ScalarTimesThis(-2./N);
        return *result;
        
    };
    
    Vector &GrassPCA::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
    {
        //std::cout << A.Getcol() << "------" << A.Getrow() << std::endl;
        //std::cout << etax.Getcol() << "BBBBBB" << etax.Getrow() << std::endl;
        Vector AAT(n, n), xxT(n, n);
        AAT.AlphaABaddBetaThis(1, A, GLOBAL::N, A, GLOBAL::T, 0);
        xxT.AlphaABaddBetaThis(1, x, GLOBAL::N, x, GLOBAL::T, 0);
        result->AlphaABaddBetaThis(1, AAT, GLOBAL::N, etax, GLOBAL::N, 0);
        result->AlphaABaddBetaThis(-1, xxT, GLOBAL::N, *result, GLOBAL::N, 1);
        result->ScalarTimesThis(-2./N);
        return *result;
    };
}; /*end of ROPTLIB namespace*/
