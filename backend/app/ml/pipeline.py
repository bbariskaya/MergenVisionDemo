from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings, settings
from app.ml.alignment import align_face
from app.ml.postprocess import Detection, decode_detections
from app.ml.preprocess import load_image, preprocess_detector, preprocess_recognizer
from app.ml.trt_engine import TrtEngine


@dataclass
class FaceExtraction:
    bbox: np.ndarray
    landmarks: np.ndarray
    embedding: np.ndarray


def l2_normalize(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(x))
    if norm == 0:
        raise ValueError("zero embedding cannot be normalized")
    return x / norm


class FacePipeline:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.detector = TrtEngine(Path(cfg.detector_engine_path))
        self.recognizer = TrtEngine(Path(cfg.embedder_engine_path))

    def warmup(self) -> None:
        # Warm once with representative shapes. This is a no-op if engines are
        # already warmed, but ensures CUDA buffers and stream are functional.
        self.detector.warmup(
            {"input.1": (1, 3, self.cfg.detector_input_size, self.cfg.detector_input_size)}
        )
        self.recognizer.warmup({"input.1": (1, 3, 112, 112)})

    def close(self) -> None:
        self.detector.close()
        self.recognizer.close()

    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]:
        tensor, scale = preprocess_detector(image, self.cfg.detector_input_size)
        outputs = self.detector.infer({"input.1": tensor})
        detections = decode_detections(
            outputs,
            input_size=self.cfg.detector_input_size,
            conf_threshold=self.cfg.detector_confidence_threshold,
            nms_threshold=self.cfg.detector_nms_iou,
        )
        return detections, scale

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        tensor = preprocess_recognizer(aligned)
        outputs = self.recognizer.infer({"input.1": tensor})
        embedding = list(outputs.values())[0].reshape(-1)
        return l2_normalize(embedding)

    def _recognizer_max_batch(self) -> int:
        # Derive the maximum batch size from the engine's optimization profile.
        # The recognizer input name is input.1 and its first dimension is N.
        try:
            _, _, max_shape = self.recognizer.engine.get_tensor_profile_shape(
                "input.1", 0
            )
            return int(max_shape[0])
        except Exception:
            return 64

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        max_batch = self._recognizer_max_batch()
        chunks = [
            aligned_faces[i : i + max_batch]
            for i in range(0, len(aligned_faces), max_batch)
        ]
        all_embeddings: list[np.ndarray] = []
        for chunk in chunks:
            batch = np.stack(
                [preprocess_recognizer(face)[0] for face in chunk], axis=0
            )
            outputs = self.recognizer.infer({"input.1": batch})
            embeddings = list(outputs.values())[0]
            all_embeddings.append(embeddings)
        combined = np.concatenate(all_embeddings, axis=0)
        if not np.isfinite(combined).all():
            raise ValueError("batched recognizer output contains non-finite values")
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("batched recognizer output contains zero-norm embedding")
        return combined / norms

    def extract(self, image: np.ndarray) -> list[FaceExtraction]:
        detections, scale = self.detect(image)
        if not detections:
            return []

        inv_scale = 1.0 / scale
        aligned_faces: list[np.ndarray] = []
        scaled_bboxes: list[np.ndarray] = []
        scaled_landmarks: list[np.ndarray] = []
        for det in detections:
            aligned_faces.append(align_face(image, det.landmarks * inv_scale))
            scaled_bboxes.append(det.bbox * inv_scale)
            scaled_landmarks.append(det.landmarks * inv_scale)

        embeddings = self.embed_batch(aligned_faces)

        results: list[FaceExtraction] = []
        for bbox, landmarks, embedding in zip(
            scaled_bboxes, scaled_landmarks, embeddings, strict=True
        ):
            results.append(
                FaceExtraction(
                    bbox=bbox,
                    landmarks=landmarks,
                    embedding=l2_normalize(embedding),
                )
            )
        return results

    def extract_from_path(self, path: str | Path) -> list[FaceExtraction]:
        image = load_image(str(path))
        return self.extract(image)
