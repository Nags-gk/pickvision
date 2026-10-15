"""Model input preprocessing shared by training and serving (numpy only, no PyTorch)."""
import numpy as np

CAM_HEIGHT = 1.0
HEIGHT_RANGE = 0.3  # metres of pile height mapped to [0, 1]


def height_channel(depth: np.ndarray) -> np.ndarray:
    """Depth (m, 0 = missing) -> normalised height above the tote floor (0 where missing)."""
    h = np.clip((CAM_HEIGHT - depth) / HEIGHT_RANGE, 0, 1)
    h[depth <= 0] = 0
    return h.astype("float32")
