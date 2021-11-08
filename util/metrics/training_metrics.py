import torch as t


def tp_score_global(y: t.Tensor, pred: t.Tensor, pred_threshold=0.5):
    """
    True Positive accuracy
    :param y: ground truth
    :param pred: prediction of same shape
    :param pred_threshold: the pred_threshold at what probability a class is considered "True"
    :return: The True Positive sum/score
    """
    mask = y >= pred_threshold
    return t.sum(pred[mask] >= pred_threshold) / t.sum(mask)


def fp_score_global(y: t.Tensor, pred: t.Tensor, pred_threshold=0.5):
    """
    False Positive accuracy
    :param y: ground truth
    :param pred: prediction of same shape
    :param pred_threshold: the pred_threshold at what probability a class is considered "True"
    :return: The True Positive sum/score
    """
    mask = y < pred_threshold
    return t.sum(pred[mask] >= pred_threshold) / t.sum(mask)


def tn_score_global(y: t.Tensor, pred: t.Tensor, pred_threshold=0.5):
    """
    :param y: ground truth
    :param pred: prediction of same shape
    :param pred_threshold: the pred_threshold at what probability a class is considered "True"
    :return: The True Positive sum/score
    """
    mask = y < pred_threshold
    return t.sum(pred[mask] < pred_threshold) / t.sum(mask)


def fn_score_global(y: t.Tensor, pred: t.Tensor, pred_threshold=0.5):
    """
    :param y: ground truth
    :param pred: prediction of same shape
    :param pred_threshold: the pred_threshold at what probability a class is considered "True"
    :return: The True Positive score
    """
    mask = y >= pred_threshold
    return t.sum(pred[mask] < pred_threshold) / t.sum(mask)


def tp_score_label(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    """
    Returns the TP score for each class in a vector 
    :param y: 
    :param pred: 
    :param pred_threshold: 
    :return: 
    """
    mask = (y >= y_threshold)
    return t.sum(t.where(mask, pred, t.tensor(-0.1, dtype=pred.dtype, device=pred.device)) >= pred_threshold,
                 dim=0) / t.sum(mask, dim=0)


def fp_score_label(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    """
    Returns the TP score for each class in a vector 
    :param y: 
    :param pred: 
    :param pred_threshold: 
    :return: 
    """
    mask = y < y_threshold
    return t.sum(t.where(mask, pred, t.tensor(-0.1, dtype=pred.dtype, device=pred.device)) >= pred_threshold,
                 dim=0) / t.sum(mask, dim=0)


def tn_score_label(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    """
    Returns the TP score for each class in a vector 
    :param y: 
    :param pred: 
    :param pred_threshold: 
    :return: 
    """
    mask = y < y_threshold
    return t.sum(t.where(mask, pred, t.tensor(1.1, dtype=pred.dtype, device=pred.device)) < pred_threshold,
                 dim=0) / t.sum(mask, dim=0)


def fn_score_label(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    """
    Returns the TP score for each class in a vector 
    :param y: 
    :param pred: 
    :param pred_threshold: 
    :return: 
    """
    mask = y >= y_threshold
    return t.sum(t.where(mask, pred, t.tensor(1.1, dtype=pred.dtype, device=pred.device)) < pred_threshold,
                 dim=0) / t.sum(mask, dim=0)


def micro_avg_precision_score(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    tps = tp_score_label(y, pred, y_threshold, pred_threshold)
    fps = fp_score_label(y, pred, y_threshold, pred_threshold)
    return t.nansum(tps) / t.nansum(tps + fps)


def micro_avg_recall_score(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    tps = tp_score_label(y, pred, y_threshold, pred_threshold)
    fns = fn_score_label(y, pred, y_threshold, pred_threshold)
    return t.nansum(tps) / (t.nansum(tps) + t.nansum(fns))


def f1_score(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    precision = micro_avg_precision_score(y, pred, y_threshold, pred_threshold)
    recall = micro_avg_recall_score(y, pred, y_threshold, pred_threshold)
    return 2 * precision * recall / (precision + recall)


def accuracy(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    tps = tp_score_label(y, pred, y_threshold, pred_threshold)
    fps = fp_score_label(y, pred, y_threshold, pred_threshold)
    tns = tn_score_label(y, pred, y_threshold, pred_threshold)
    fns = fn_score_label(y, pred, y_threshold, pred_threshold)
    return (t.nansum(tps) + t.nansum(tns)) / (t.nansum(fps) + t.nansum(fns) + t.nansum(tps) + t.nansum(tns))


def class_count_prediction(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    return t.sum(pred > pred_threshold, dim=0)


def class_count_truth(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5):
    return t.sum(y, dim=0)


def zero_fit_score(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5) -> float:
    mask = y != 0.0
    inverse_mask = ~mask
    return 1.0 - t.sqrt(t.sum(t.square(y[inverse_mask] - pred[inverse_mask]))) / t.sum(inverse_mask)  # zero fit goal


def class_fit_score(y: t.Tensor, pred: t.Tensor, y_threshold=0.5, pred_threshold=0.5) -> float:
    mask = y != 0.0
    return 1.0 - t.sqrt(t.sum(t.square(y[mask] - pred[mask]))) / t.sum(mask)  # zero fit goal
