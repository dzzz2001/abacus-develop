
#include "Solvers/RSVRG.h"

/*Define the namespace*/
namespace ROPTLIB{
    
    RSVRG::RSVRG(const Problem *prob, const Variable *initialx)
    {
        Initialization(prob, initialx);
    };
    
    void RSVRG::CheckParams(void)
    {
        SolversSMSVRG::CheckParams();
        
        printf("STOCHASTIC VARIANCE REDUCTION TYPE METHODS PARAMETERS: NONE\n");
    };

    void RSVRG::SetProbX(const Problem *prob, const Variable *initialx)
    {
        SolversSMSVRG::SetProbX(prob, initialx);
        prob->SetUseGrad(true);
        prob->SetUseHess(false);
    };
    
    void RSVRG::SetDefaultParams(void)
    {
        SolversSMSVRG::SetDefaultParams();
        SolverName.assign("RSVRG");
    };
    
    void RSVRG::GetSearchDir(void)
    {
        Mani->ScalarTimesVector(x1, -1, gf1, &eta1);
    };
}; /*end of ROPTLIB namespace*/
