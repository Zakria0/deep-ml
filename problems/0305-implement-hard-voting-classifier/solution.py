def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    final = []
    for i in range(len(predictions[0])):
        classes, counts = dict(), []
        for j in range(len(predictions)):
            pred = predictions[j][i]
            if pred in classes:
                counts[classes[pred]] += 1
            else:
                classes[pred] = len(classes)
                counts.append(1)
        
        max_class, max_count = None, float('-inf')
        for (cls, idx) in classes.items():
            if counts[idx] > max_count:
                max_class = cls
                max_count = counts[idx]
            if counts[idx] == max_count and cls < max_class:
                max_class = cls
        
        final.append(max_class)
    
    return final







