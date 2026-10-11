import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.
    
    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix under the policy
        R: np.ndarray of shape (n_states,), expected immediate reward for each state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'
    
    Returns:
        np.ndarray: updated value function after n_sweeps
    """
    if method == 'synchronous':
        for _ in range(n_sweeps):
            V_new = V.copy()
            for state in range(len(V)):
                V_new[state] = R[state] + gamma * np.sum(P[state] @ V)
            V = V_new.copy()
        return V_new
    else:
        for _ in range(n_sweeps):
            for state in range(len(V)):
                V[state] = R[state] + gamma * np.sum(P[state] @ V)
        
        return V
    



