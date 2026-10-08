import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """
    np.random.seed(seed)
    q_values = np.asarray(q_values)

    e = np.exp((q_values - np.max(q_values)) / temperature)
    probs = e / np.sum(e)
    
    return probs.tolist(), np.random.choice(len(q_values), p=probs)
    