from metrics.utils.utils import get_metrics_variables

def recall(y_true, y_pred):
    '''Calculate recall'''
    tp, _, _, fn = get_metrics_variables(y_true, y_pred)
    recall = tp / (tp + fn)

    return recall