def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    counts = dict()
    for x in data:
        if x[1] not in counts:
            counts[x[1]] = [1, x[0]]
        else:
            counts[x[1]][0] += 1
            counts[x[1]].append(x[0])
    
    n_min, min_c = float('inf'), None
    for c in counts:
        if counts[c][0] < n_min:
            n_min = counts[c][0]
            min_c = c
        
    undersampled_data = []
    for c in counts:
        for x in counts[c][1: n_min+1]:
            undersampled_data.append((x, c))
    
    return sorted(undersampled_data)













