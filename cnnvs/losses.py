from metrics import dsc as _dsc


def dsc(y_true, y_pred):
    return -_dsc(y_true, y_pred, 1.0)
