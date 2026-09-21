import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    n = len(y_true)
    y_scores = np.asarray(y_scores)

    precisions = []
    recalls = []
    thresholds = []
    for t in y_scores:
        pred = np.int32(y_scores >= t)

        TP = np.sum([1 for i in range(n) if pred[i] == y_true[i] == 1])
        FP = np.sum([1 for i in range(n) if pred[i] != y_true[i] == 0])
        FN = np.sum([1 for i in range(n) if pred[i] != y_true[i] == 1])

        if not sum(pred):
            precisions.append(1.)
        else:
            precisions.append(float(TP / (TP + FP)))
        
        if not sum(y_true):
            recalls.append(0.)
        else:
            recalls.append(float(TP / (TP + FN)))

        thresholds.append(float(t))

    return precisions, recalls, thresholds



