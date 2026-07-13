import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def resize_image(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = np.array(
        Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
    )
    # Pad to square; content stays at top-left, padding at bottom/right
    pad_h = size - new_h
    pad_w = size - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return padded, scale


def _to_nchw_contiguous(
    arr: np.ndarray, *, mean: float, std: float
) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def preprocess_detector(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    resized, scale = resize_image(image, input_size)
    # Official SCRFD contract: RGB, mean=127.5, std=128.0
    arr = _to_nchw_contiguous(resized, mean=127.5, std=128.0)
    tensor = np.ascontiguousarray(np.expand_dims(arr, 0), dtype=np.float32)
    assert tensor.flags["C_CONTIGUOUS"]
    return tensor, scale


def preprocess_recognizer(aligned_face: np.ndarray) -> np.ndarray:
    # Official ArcFace glintr100 contract: RGB, mean=127.5, std=127.5
    arr = _to_nchw_contiguous(aligned_face, mean=127.5, std=127.5)
    tensor = np.ascontiguousarray(np.expand_dims(arr, 0), dtype=np.float32)
    assert tensor.flags["C_CONTIGUOUS"]
    return tensor
