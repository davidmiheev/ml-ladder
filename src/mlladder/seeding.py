"""One seeding entry point, so every rung reports reproducible numbers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed ``random``, ``numpy`` and — if installed — ``torch``.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed
