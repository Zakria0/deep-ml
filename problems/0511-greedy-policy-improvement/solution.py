def greedy_policy_improvement(V: dict, transitions: dict, gamma: float) -> tuple:
    """
    Perform greedy policy improvement given a value function and MDP model.
    
    Args:
        V: dict mapping state -> float (state-value function)
        transitions: dict mapping (state, action) -> list of (probability, next_state, reward)
        gamma: float, discount factor
    
    Returns:
        tuple: (policy, Q)
            policy: dict mapping state -> best action
            Q: dict mapping (state, action) -> float
    """
    Q = {}
    for pair in transitions.keys():
        s = 0
        for triple in transitions[pair]:
            s += triple[0] * (triple[2] + gamma * V[triple[1]])
        Q[pair] = s
    
    policy = {}
    for pair in Q.keys():
        a = policy.get(pair[0], None)
        if a is None or Q[(pair[0], a)] < Q[pair]:
            policy[pair[0]] = pair[1]
    
    return policy, Q





