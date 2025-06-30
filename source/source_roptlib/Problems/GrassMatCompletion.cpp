
#include "Problems/GrassMatCompletion.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    GrassMatCompletion::GrassMatCompletion(integer *inir, integer *injc, realdp *invals, integer innz, integer inN, integer inD, integer inr)
    {
        ir = inir;
        jc = injc;
        vals = invals;
        nz = innz;
        N = inN;
        d = inD;
        r = inr;
        
        NumGradHess = false;
    };
    
    GrassMatCompletion::~GrassMatCompletion(void)
    {
    };
    
    realdp GrassMatCompletion::f(const Variable &x) const
    {
        Vector batch_index(N);
        realdp *batch_indexptr = batch_index.ObtainWriteEntireData();
        for(integer i = 0; i < N; i++)
            batch_indexptr[i] = i;
        return Stof(x, batch_index);   //test problem
    };

    realdp GrassMatCompletion::Stof(const Variable &x, const Vector &batch_index) const
    {
        realdp result = 0;
        Vector UaiMinuszi;
        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            Getai(x, static_cast<integer> (batch_index[i]), &UaiMinuszi);
            result = result + UaiMinuszi.DotProduct(UaiMinuszi);
        }
        result = result/batch_index.Getlength();
        return result;
    };

    Vector &GrassMatCompletion::EucStoGrad(const Variable &x, Vector *result, const Vector &batch_index) const
    { /* Did not avoid redundant computations? --WH */
        Vector gradi(d,r);
        result->SetToZeros();
        if (batch_index.Getlength() == 0)
        {
            for (integer i = 0; i < N; i++)
            {
                GetGradi(x, i, &gradi);
                result->AlphaXaddThis(1, gradi);
            }
            result->ScalarTimesThis(2./N);
            return *result;
        }

        for (integer i = 0; i < batch_index.Getlength(); i++)
        {
            GetGradi(x, static_cast<integer> (batch_index[i]), &gradi);
            result->AlphaXaddThis(1, gradi);
        }
        result->ScalarTimesThis(2./batch_index.Getlength());
        return *result;
    };

    Vector &GrassMatCompletion::EucGrad(const Variable &x, Vector *result) const
    {
        Vector batch_index(N);
        realdp *batch_indexptr = batch_index.ObtainWriteEntireData();
        for(integer i = 0; i < N; i++)
            batch_indexptr[i] = i;
        
        return EucStoGrad(x, result, batch_index);
    };
    
//    Vector &GrassMatCompletion::EucHessianEta(const Variable &x, const Vector &etax, Vector *result) const
//    {
//        Vector gradi(d, r);
//        Vector egrad(d, r);
//        Vector temp1(d, r);    // store 2(eta a_i - z_i) * a_i^T
//        Vector temp2(d, r);
//        Vector xTgrad(r, r);
//        result->SetToZeros();
//        egrad.SetToZeros();
//        temp2.SetToZeros();
//        for (integer i = 0; i < N; i++)
//        {
//            GetGradi(x, i, &gradi);
//            GetGradi(etax, i, &temp1);
//            egrad.AlphaXaddThis(1, gradi);
//            temp2.AlphaXaddThis(1, temp1);
//        }
//        xTgrad.AlphaABaddBetaThis(1, x, GLOBAL::T, egrad, GLOBAL::N, 0);
//        result->AlphaABaddBetaThis(-1, etax, GLOBAL::N, xTgrad, GLOBAL::T, 0);
//        result->AlphaXaddThis(2, temp2);
//        result->ScalarTimesThis(1./N);
//        
//        std::cout << "nresult:" << result->Fnorm() << ":" << etax.Fnorm() << ":" << temp2.Fnorm() << ":" << xTgrad.Fnorm() << std::endl;//---
//        return *result;
//    };
    
    Vector &GrassMatCompletion::Getai(const Variable &x, const integer column, Vector *result) const
    {
        integer rowsize = 0;
        integer tmpr;
        integer tmpj;
        integer *rowdex = new integer[d];
        integer *jdex = new integer[d];
        for (integer j = 0; j < nz; j++)
        {
            if (jc[j] == column)
            {
                rowdex[rowsize] = ir[j];  // ir value when jc is chosen
                jdex[rowsize] = j;    // index of picked point in ir,jc,vals
                for (integer k = rowsize; k > 0; k--)
                {
                    if (rowdex[k] < rowdex[k-1]) // guarantee ir is in increasing order
                    {
                        tmpr = rowdex[k-1];
                        rowdex[k-1] = rowdex[k];
                        rowdex[k] = tmpr;
                        tmpj = jdex[k-1];
                        jdex[k-1] = jdex[k];
                        jdex[k] = tmpj;
                    }
                }
                rowsize++;
            }
        }
        Vector Ustar(rowsize,r); // U_i_{\Omega_row}
        Vector zstar(rowsize,1); // z_i_{\Omega_row}
        Vector Urow(1,r);
        realdp *Ustarptr = Ustar.ObtainWriteEntireData();
        realdp *zstarptr = zstar.ObtainWriteEntireData();
        for (integer j = 0; j < rowsize; j++)
        {
            zstarptr[j] = vals[jdex[j]];
            Urow = x.GetSubmatrix(rowdex[j], rowdex[j], 0, r-1);
            const realdp *Urowptr = Urow.ObtainReadData();
            for (integer k = 0; k < r; k++)
            {
                Ustarptr[j+k*rowsize] = Urowptr[k];
            }
        }
        Vector UTU(r,r);
        Vector UTz(r);
        Vector identity(r,r); identity.SetToIdentity();
        Vector ai(r);
        UTU.AlphaABaddBetaThis(1, Ustar, GLOBAL::T, Ustar, GLOBAL::N, 0);
        UTz.AlphaABaddBetaThis(1, Ustar, GLOBAL::T, zstar, GLOBAL::N, 0);
        ai.AlphaABaddBetaThis(1, UTU.PseudoInverseMatrix(), GLOBAL::N, UTz, GLOBAL::N, 0); // ai = (U^TU)^{-1}U^Tz
//        ai.AlphaABaddBetaThis(1, identity/UTU, GLOBAL::N, UTz, GLOBAL::N, 0); // ai = (U^TU)^{-1}U^Tz
//        if(column == 740)
//        {
//            (x.GetTranspose() * x).Print("UTUall:");//--
//            UTU.Print("UTU:");//--
//            (identity/UTU).Print("(U^T U)^{-1}:");//---
//        }
        *result = Vector(rowsize);
        result->AlphaABaddBetaThis(1, Ustar, GLOBAL::N, ai, GLOBAL::N, 0); // Uai = Ustar * ai
//        if(column == 740)
//            std::cout << "h1:" << Ustar.Fnorm() << ":" << ai.Fnorm() << std::endl;//--
//        if(column == 740)
//            std::cout << "h2:" << result->Fnorm() << std::endl;//--
        result->AlphaXaddThis(-1,zstar);
//        if(column == 740)
//            std::cout << "h3:" << result->Fnorm() << std::endl;//--
        delete[] rowdex;
        delete[] jdex;
        return *result;
    };
    
    Vector &GrassMatCompletion::GetGradi(const Variable &x, const integer column, Vector *result) const
    {
        integer rowsize = 0;
        integer tmpr;
        integer tmpj;
        integer *rowdex = new integer[d];
        integer *jdex = new integer[d];
        for (integer j = 0; j < nz; j++)
        {
            if (jc[j] == column)
            {
                rowdex[rowsize] = ir[j];  // ir value when jc is chosen
                jdex[rowsize] = j;    // # of picked point in ir,jc,vals
                for (integer k = rowsize; k > 0; k--)
                {
                    if (rowdex[k] < rowdex[k-1]) // increasing
                    {
                        tmpr = rowdex[k-1];
                        rowdex[k-1] = rowdex[k];
                        rowdex[k] = tmpr;
                        tmpj = jdex[k-1];
                        jdex[k-1] = jdex[k];
                        jdex[k] = tmpj;
                    }
                }
                rowsize++;
            }
        }
        Vector Ustar(rowsize,r); // U_i_{\Omega_row}
        Vector zstar(rowsize,1); // z_i_{\Omega_row}
        Vector Urow(1,r);
        realdp *Ustarptr = Ustar.ObtainWriteEntireData();
        realdp *zstarptr = zstar.ObtainWriteEntireData();
        for (integer j = 0; j < rowsize; j++)
        {
            zstarptr[j] = vals[jdex[j]];
            Urow = x.GetSubmatrix(rowdex[j], rowdex[j], 0, r-1);
            const realdp *Urowptr = Urow.ObtainReadData();
            for (integer k = 0; k < r; k++)
            {
                Ustarptr[j+k*rowsize] = Urowptr[k];
            }
        }
        Vector UTU(r,r);
        Vector UTz(r);
        Vector identity(r,r); identity.SetToIdentity();
        Vector ai(r);
        Vector Uai(rowsize);
        UTU.AlphaABaddBetaThis(1, Ustar, GLOBAL::T, Ustar, GLOBAL::N, 0);
        UTz.AlphaABaddBetaThis(1, Ustar, GLOBAL::T, zstar, GLOBAL::N, 0);
        ai.AlphaABaddBetaThis(1, identity/UTU, GLOBAL::N, UTz, GLOBAL::N, 0);
        Uai.AlphaABaddBetaThis(1, Ustar, GLOBAL::N, ai, GLOBAL::N, 0);
        Uai.AlphaXaddThis(-1,zstar);  // Let Uai work as (Uai - zi)_{\Omega_i}
        result->SetToZeros();
        realdp *resultptr = result->ObtainWriteEntireData();
        const realdp *UaiMinusziptr = Uai.ObtainReadData();
        const realdp *aiptr = ai.ObtainReadData();
        for (integer j = 0; j < rowsize; j++)
        {
            for (integer k = 0; k < r; k++)
            {
                resultptr[rowdex[j] + k*d] = UaiMinusziptr[j] * aiptr[k];
            }
        }
        delete[] rowdex;
        delete[] jdex;
        return *result;
    };
}; /*end of ROPTLIB namespace*/
