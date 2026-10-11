def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    states  = list(policy.keys())
    rows = max(state[0] for state in states) + 1
    cols = max(state[1] for state in states) + 1

    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    terminals = {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}

    def step(state, action):
        dr, dc = moves[action]
        r, c = state[0] + dr, state[1] + dc
        return (r, c) if 0 <= r < rows and 0 <= c < cols else state

    V = [[0.] * cols for _ in range(rows)]
    while True:
        delta = 0
        
        new_V = [row[:] for row in V]
        for state in states:
            if state in terminals:
                continue

            new_v = 0.0
            for action, prob in policy[state].items():
                nr, nc = step(state, action)
                new_v += prob * (-1 + gamma * V[nr][nc])

            delta = max(delta, abs(new_v - V[state[0]][state[1]]))
            new_V[state[0]][state[1]] = new_v

        V = new_V    
        if delta < threshold:
            break
    
    return V
