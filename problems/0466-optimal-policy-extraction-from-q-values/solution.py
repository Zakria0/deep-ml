import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)
    """
    optimal_actions = np.argmax(Q, axis=1)
    state_values    = np.max(Q, axis=1)
    advantages      = np.round((Q.T - state_values).T, 4)
    return {
        'optimal_actions': optimal_actions.tolist(),
        'state_values'   :    state_values.tolist(),
        'advantages'     :      advantages.tolist()
    }