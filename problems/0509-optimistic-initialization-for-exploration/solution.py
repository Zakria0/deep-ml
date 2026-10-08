import numpy as np

def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """
    n = len(true_rewards)
    Q = np.array([initial_q] * n)
    N = np.zeros(n)

    for i in range(n_steps):
        action = np.argmax(Q)
        N[action] += 1
        Q[action] += step_size * (true_rewards[action] - Q[action])
    
    return Q, N









