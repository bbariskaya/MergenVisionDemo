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


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    landmarks = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARC_FACE_SRC)
    if M is None:
        raise ValueError("Could not estimate affine transform for alignment")
    aligned = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
    return aligned
