import numpy as np
from Global_Vars import Global_Vars
from Model_TLACNN import Model_TLACNN


def Obj_fun_CLS(Soln):
    Data = Global_Vars.Data
    Target = Global_Vars.Target
    RAN_weight = Global_Vars.RAN_weight
    Dense_weight = Global_Vars.Dense_weight
    Fitn = np.zeros(Soln.shape[0])
    Batch_size = 16
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            Eval = Model_TLACNN(RAN_weight, Dense_weight, Data, Target, Batch_size, sol)
            Fitn[i] = (1 / Eval[4]) + Eval[12]
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        Eval = Model_TLACNN(RAN_weight, Dense_weight, Data, Target, Batch_size, sol)
        Fitn = (1 / Eval[4]) + Eval[12]
        return Fitn



