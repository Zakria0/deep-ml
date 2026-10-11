import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers
        actions: List of action identifiers
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a)
        rewards: Dict mapping (s, a, s') -> R(s,a,s')
        policy: Dict mapping (s, a) -> pi(a|s)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places
    """
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        old_q = Q.copy()
        
        for s in states:
            for a in actions:
                Q[s, a] = float(np.sum([transition_probs.get((s, a, s_prime), 0) * (rewards.get((s, a, s_prime), 0) + gamma * np.sum([policy[s_prime, a_prime] * Q[s_prime, a_prime] for a_prime in actions])) for s_prime in states]))

        if np.max([np.abs(Q[s, a] - old_q[s, a]) for s in states for a in actions]) < theta:
            break
        
    return {(s, a): round(Q[s, a], 4) for (s, a) in Q}

    