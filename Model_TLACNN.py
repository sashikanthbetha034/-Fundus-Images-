from Classificaltion_Evaluation import ClassificationEvaluation
from Model_Densenet import Model_Dil_Densenet_cls
from Models_Ran import Model_Ran_cls


def Model_TLACNN(RAN_weight, Dense_weight, data, Tar, Batch_size, sol=None):
    if sol is None:
        sol = [-20, 20]
    eval, Pred_mlp = Model_Ran_cls(RAN_weight, data, Tar, Batch_size, sol)
    eval, Pred_Caps = Model_Dil_Densenet_cls(Dense_weight, data, Tar, Batch_size,  sol)

    pred = (Pred_mlp + Pred_Caps)/2

    Eval = ClassificationEvaluation(pred, Tar)

    return Eval, pred
