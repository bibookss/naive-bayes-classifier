import pandas as pd

def calculate_tp(y_true, y_pred):
    '''Calculate true positive whose label is spam and predicted as spam'''
    df = pd.DataFrame({'label': y_true, 'prediction': y_pred})
    tp = len(df[(df['label'] == 'spam') & (df['prediction'] == 'spam')])

    return tp

def calculate_fp(y_true, y_pred):
    '''Calculate false positive whose label is ham but predicted as spam'''
    df = pd.DataFrame({'label': y_true, 'prediction': y_pred})
    fp = len(df[(df['label'] == 'ham') & (df['prediction'] == 'spam')])

    return fp

def calculate_tn(y_true, y_pred):
    '''Calculate true negative whose label is ham and predicted as ham'''
    df = pd.DataFrame({'label': y_true, 'prediction': y_pred})
    tn = len(df[(df['label'] == 'ham') & (df['prediction'] == 'ham')])

    return tn

def calculate_fn(y_true, y_pred):
    '''Calculate false negative whose label is spam but predicted as ham'''
    df = pd.DataFrame({'label': y_true, 'prediction': y_pred})
    fn = len(df[(df['label'] == 'spam') & (df['prediction'] == 'ham')])

    return fn

def get_metrics_variables(y_true, y_pred):
    tp = calculate_tp(y_true, y_pred)
    fp = calculate_fp(y_true, y_pred)
    tn = calculate_tn(y_true, y_pred)
    fn = calculate_fn(y_true, y_pred)

    return tp, fp, tn, fn