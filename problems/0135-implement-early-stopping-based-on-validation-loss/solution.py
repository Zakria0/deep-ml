from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    c = 0
    best_epoch = 0
    for i in range(1, len(val_losses)):
        if val_losses[best_epoch] - val_losses[i] > min_delta:
            best_epoch = i
            c = 0
        elif c < patience:
            c += 1
        else:
            break         
    return c + best_epoch, best_epoch