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
    resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR))
    # Pad to square
    pad_h = size - new_h
    pad_w = size - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return padded, scale


def preprocess_detector(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    resized, scale = resize_image(image, input_size)
    # HWC -> CHW, normalize to [-1, 1]
    arr = resized.astype(np.float32)
    arr = (arr - 127.5) / 127.5
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr, scale


def preprocess_recognizer(aligned_face: np.ndarray) -> np.ndarray:
    # aligned_face expected RGB 112x112
    arr = aligned_face.astype(np.float32)
    arr = (arr - 127.5) / 127.5
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)
