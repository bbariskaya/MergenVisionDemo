import cv2
import numpy as np

# Standard ArcFace 112x112 reference landmarks
ARC_FACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a 2x3 similarity transform (rotation, uniform scale, translation).

    The model is:
        dst_x = a * src_x - b * src_y + tx
        dst_y = b * src_x + a * src_y + ty
    which is linear in [a, b, tx, ty] and solved with ordinary least squares.
    This is deterministic and does not use RANSAC.
    """
    if src.shape != dst.shape or src.shape != (5, 2):
        raise ValueError("src and dst must be (5, 2) arrays")
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    A = np.empty((2 * len(src), 4), dtype=np.float64)
    for i, (sx, sy) in enumerate(src):
        A[2 * i] = [sx, -sy, 1.0, 0.0]
        A[2 * i + 1] = [sy, sx, 0.0, 1.0]
    b = dst.ravel()

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_param, tx, ty = x

    M = np.array(
        [[a, -b_param, tx], [b_param, a, ty]],
        dtype=np.float32,
    )
    return M


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    if size != 112:
        raise NotImplementedError(
            f"Alignment size {size} is not supported; only 112 is supported"
        )
    landmarks = landmarks.astype(np.float32)
    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks contain non-finite values")
    M = _estimate_similarity_transform(landmarks, ARC_FACE_SRC)
    aligned = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
    return aligned
