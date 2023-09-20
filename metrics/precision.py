from metrics.utils.utils import get_metrics_variables

def precision(y_true, y_pred):
    '''Calculate precision'''
    tp, fp, _, _ = get_metrics_variables(y_true, y_pred)
    precision = tp / (tp + fp)

    return precision