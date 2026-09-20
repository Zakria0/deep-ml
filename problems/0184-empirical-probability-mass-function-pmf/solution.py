def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    n = len(samples)
    occ = {}
    for i in samples:
        occ[i] = occ.get(i, 0) + 1
    
    L = sorted(list(set(samples)))

    return [(i, occ[i] / n) for i in L]