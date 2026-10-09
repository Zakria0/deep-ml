import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    next_states = list(P[state][action].keys())

    ps = np.array([P[state][action][hadik] for hadik in next_states])
    rs = np.array([R[state][action][hadik] for hadik in next_states])
    vs = V[next_states]

    return np.sum(ps * (rs + gamma * vs))


    