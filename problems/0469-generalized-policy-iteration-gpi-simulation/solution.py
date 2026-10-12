import numpy as np

def generalized_policy_iteration(
    num_states: int,
    num_actions: int,
    transitions: np.ndarray,
    rewards: np.ndarray,
    discount: float,
    num_iterations: int,
    eval_sweeps: int,
    terminal_states: list
) -> dict:
    """
    Simulate Generalized Policy Iteration on a finite MDP.
    
    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available
        transitions: Shape (S, A, S) transition probabilities
        rewards: Shape (S, A, S) reward function
        discount: Discount factor gamma in [0, 1]
        num_iterations: Number of GPI cycles (eval + improve)
        eval_sweeps: Number of synchronous policy evaluation sweeps per cycle
        terminal_states: List of terminal state indices
    
    Returns:
        Dictionary with:
            'values': List of floats (rounded to 4 decimal places)
            'policy': List of ints (action for each state)
    """
    V = np.zeros(num_states)
    policy = [0] * num_states

    for cycle in range(num_iterations):
        for sweep in range(eval_sweeps):
            new_V = V.copy()
            for s in range(num_states):
                new_V[s] = np.sum(transitions[s, policy[s]] * (rewards[s, policy[s]] + discount * V)) if s not in terminal_states else 0
            V = new_V.copy()

        for s in range(num_states):
            policy[s] = np.argmax([np.sum([transitions[s, a] * (rewards[s, a] + discount * V)]) for a in range(num_actions)])
    
    return {
        'values': list(np.round(V, 4)),
        'policy': policy
    }