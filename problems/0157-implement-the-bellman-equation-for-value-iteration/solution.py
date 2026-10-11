import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    new_V = V.copy()
    for i in range(len(transitions)):
      s = transitions[i]
      if not s: continue
      v_max = -np.inf
      for a in s:
        if not s[a]:
          continue
        q = 0
        for nuple in s[a]:
          q += nuple[0] * (nuple[2] + gamma * (1 - nuple[3]) * V[nuple[1]])

        if v_max < q:
          v_max = q
      new_V[i] = v_max

    return new_V


