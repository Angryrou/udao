import random

import numpy as np
import torch

EPS = 1e-7


def set_deterministic_torch(seed: int = 0) -> None:
    """
    Set seeds and configurations to enable deterministic behavior in PyTorch.

    Parameters
    ----------
    seed : int
        Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)  # type: ignore
