import numpy as np

def async_value_iteration(num_states: int, transitions: list, gamma: float, update_order: list) -> list:
    """
    Perform asynchronous value iteration on an MDP.
    
    Args:
        num_states: Number of states in the MDP
        transitions: transitions[s][a] = [(prob, next_state, reward), ...]
        gamma: Discount factor
        update_order: Sequence of state indices to update (in order)
    
    Returns:
        List of float values representing the value function
    """
    V = [0.0] * num_states
    for s in update_order:
        v_max = -np.inf
        for a in range(len(transitions[s])):
            q = 0
            for triple in transitions[s][a]:
                p = triple[0]
                r = triple[2]
                q += p * (r + gamma * V[triple[1]])
            
            if q > v_max:
                v_max = q
        V[s] = v_max
    
    return V



