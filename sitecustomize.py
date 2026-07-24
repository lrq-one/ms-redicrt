from __future__ import annotations

import os
import random


seed_text = os.environ.get(
    "MS2_GLOBAL_SEED"
)

if seed_text is not None:
    seed = int(seed_text)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )
    except Exception:
        pass
