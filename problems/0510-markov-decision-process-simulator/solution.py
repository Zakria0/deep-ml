import numpy as np

def simulate_mdp(P, R, policy, start_state, terminal_states, gamma, num_episodes, max_steps, seed=42):
    """
    Simulate episodes in an MDP and compute discounted returns.
    
    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states)
        R: Rewards, shape (n_states, n_actions, n_states)
        policy: Stochastic policy, shape (n_states, n_actions)
        start_state: Initial state for each episode
        terminal_states: List of terminal state indices
        gamma: Discount factor
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (episode_returns, average_return)
    """
    np.random.seed(seed)
    n_states, n_actions, n_states = P.shape

    episode_returns = []
    for ep in range(num_episodes):
        s = start_state
        episode_return = 0

        t = 0
        while t < max_steps:
            if s in terminal_states:
                break
            prev_s = s
            a = np.random.choice(range(n_actions), p=policy[s])
            s = np.random.choice(range(n_states), p=P[prev_s, a])
            r = R[prev_s, a, s]

            episode_return += gamma**t * r
            t += 1
        
        episode_returns.append(round(episode_return, 4))
    
    return [float(x) for x in episode_returns], round(float(np.mean(episode_returns)), 4)
            



















