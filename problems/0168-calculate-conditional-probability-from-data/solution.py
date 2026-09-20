def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    numer = 0
    denom = 0

    for elem in data:
      if elem[0] == x:
        denom += 1
        if elem[1] == y:
          numer += 1

    if not denom:
      return 0
    return numer / denom