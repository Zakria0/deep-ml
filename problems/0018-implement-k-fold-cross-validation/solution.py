import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    idx = np.arange(0, n_samples)
    split = [n_samples // k * i for i in range(k)]
    
    remainder = n_samples % k

    if not remainder:
        for i in range(remainder):
            split[i] += 1
    
    if shuffle:
        np.random.shuffle(idx)
    
    k_folds = []

    for i in range(k-1):
        idx_test  = idx[split[i]: split[i+1]]
        idx_train = np.concatenate((idx[:split[i]], idx[split[i+1]:]))
        k_folds.append((list(idx_train), list(idx_test)))
    
    k_folds.append((list(idx[:split[-1]]), list(idx[split[-1]:])))
    return k_folds



