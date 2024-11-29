
import math
from typing import Union

# In given y, 1 is positive and 0 is negative


def get_predres(
        y_pred: list, 
        y_target: list
        ):
    '''Get the global result from the comparison between predicted label and target label
    
    :param y_pred: The list of predicted labels (0 for negative and 1 for positive)
    :type y_pred: list 

    :param y_target: The list of target labels (0 for negative and 1 for positive)
    :type y_target: list 
    
    :return: Number of True Positive, Number of False Positive, Number of True Netative, 
    Number of False Negative
    :rtype: (int, int, int, int)
    '''

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_target[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_target[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_target[i] == 0:
            tn += 1
        elif y_pred[i] == 0 and y_target[i] == 1:
            fn += 1
    return tp, fp, tn, fn


def get_recall(
        tp: Union[int, float], 
        fn: Union[int, float]
        ):
    '''Get the recall value from the given True Positive and False Negative values
    
    :param tp: The number of True Positive
    :type tp: int | float 

    :param fn: The number of False Negative
    :type fn: int | float 

    :return: The Recall value as recall = tp/(tp+fn)
    :rtype: float
    '''

    if tp+fn == 0:
        return 0
    else:
        return tp/(tp+fn)



def get_precision(
        tp: Union[int, float], 
        fp: Union[int, float]
        ):
    '''Get the precision value from the given True Positive and False Positive values
    
    :param tp: The number of True Positive
    :type tp: int | float 

    :param fp: The number of False Positive
    :type fp: int | float 

    :return: The Precision value as precision = tp/(tp+fp)
    :rtype: float
    '''

    if tp+fp == 0:
        return 0
    else:
        return tp/(tp+fp)



def get_F1score(
        recall: float, 
        precision: float
        ):
    '''Get the F1 score from the given recall and precision values
    
    :param recall: The Recall value as recall = tp/(tp+fn)
    :type recall: float 

    :param precision: The Precision value as precision = tp/(tp+fp)
    :type precision: float 

    :return: The F1 score as F1 score = 2 * ((precision * recall) / (precision + recall))
    :rtype: float
    '''

    if precision+recall == 0:
        return 0
    else:
        return 2*((precision*recall)/(precision+recall))


def get_MCCscore(
        tp: Union[int, float], 
        fp: Union[int, float], 
        tn: Union[int, float], 
        fn: Union[int, float]
        ):
    '''Get MCC score from given prediction result (TP, FP, TN, FN)
    
    :param tp: Number of True Positive
    :type tp: int | float

    :param fp: Number of False Positive
    :type fp: int | float 

    :param tn: Number of True Netative
    :type tn: int | float 

    :param fn: Number of False Negative
    :type fn: int | float 

    :return: The MCC score 
    as mcc = ((tp * tn) - (fp * fn)) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    :rtype: float
    '''
    if tp == 0 and fp == 0:
        if fn == 0 and tn != 0:
            return 1
        elif fn != 0 and tn != 0:
            return 0
        elif fn != 0 and tn == 0:
            return -1
    elif tp == 0 and fn == 0:
        if fp == 0 and tn != 0:
            return 1
        if fp != 0 and tn != 0:
            return 0
        if fp != 0 and tn == 0:
            return -1
    elif tn == 0 and fp == 0:
        if tp != 0 and fn == 0:
            return 1
        elif tp != 0 and fn != 0:
            return 0
        elif tp == 0 and fn != 0:
            return -1
    elif tn == 0 and fn == 0:
        if tp != 0 and fp != 0:
            return 1
        elif tp != 0 and fp != 0:
            return 0
        elif tp == 0 and fp != 0:
            return -1
    else:
        return ((tp*tn) - (fp*fn)) / (math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)))



def get_metrics(
        y_pred: list, 
        y_target: list
        ):
    '''Get the Recall, Precision and F1 score values based on the given predicted and target 
    label lists
    
    :param y_pred: The list of predicted labels (0 for negative and 1 for positive)
    :type y_pred: list 

    :param y_target: The list of target labels (0 for negative and 1 for positive) 
    :type y_target: list 

    :return: The Recall value, The Precision value, The F1 score, and The MCC score 
    :rtype: (float, float, float, float)
    '''

    tp, fp, tn, fn = get_predres(y_pred, y_target)
    recall = get_recall(tp, fn)
    precision = get_precision(tp, fp)
    f1_score = get_F1score(recall, precision)
    mcc_score = get_MCCscore(tp, fp, tn, fn)

    return recall, precision, f1_score, mcc_score