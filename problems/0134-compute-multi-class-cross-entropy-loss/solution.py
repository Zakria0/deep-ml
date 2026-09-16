import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    losses = np.array([np.sum(-true_labels[i] * np.log(predicted_probs[i] + epsilon)) for i in range(len(true_labels))])
    return np.mean(losses)