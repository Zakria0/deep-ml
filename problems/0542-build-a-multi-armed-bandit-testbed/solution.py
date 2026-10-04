import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    
    Args:
        k: Number of arms
        num_pulls: Number of times to pull each arm
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_values, sample_means, optimal_arm)
    """
    np.random.seed(seed)

    true_values = np.random.randn(k)
    bandit = np.array([a + np.random.randn(num_pulls) for a in true_values])
    sample_means = np.mean(bandit, axis=1)
    optimal_arm = int(np.argmax(true_values))

    return (
        [round(float(v), 4) for v in true_values],
        [round(float(m), 4) for m in sample_means],
        optimal_arm
    )

