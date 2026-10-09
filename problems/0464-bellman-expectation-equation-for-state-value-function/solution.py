import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    num_states, num_actions, _ = P.shape
    p_pi = np.zeros((num_states, num_states))
    r_pi = np.zeros(num_states)

    for s in range(num_states):
        for a in range(num_actions):
            pr = 0
            for sd in range(num_states):
                pr += P[s, a, sd] * R[s, a, sd]
                p_pi[s, sd] += policy[s, a] * P[s, a, sd]
            r_pi[s] += policy[s, a] * pr

    return np.linalg.inv(np.eye(num_states) - gamma * p_pi) @ r_pi

