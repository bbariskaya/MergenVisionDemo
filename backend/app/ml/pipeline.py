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
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


class FacePipeline:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.detector = TrtEngine(Path(cfg.detector_engine_path))
        self.recognizer = TrtEngine(Path(cfg.embedder_engine_path))

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

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        batch = np.stack(
            [preprocess_recognizer(face)[0] for face in aligned_faces], axis=0
        )
        outputs = self.recognizer.infer({"input.1": batch})
        embeddings = list(outputs.values())[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def extract(self, image: np.ndarray) -> list[FaceExtraction]:
        detections, scale = self.detect(image)
        results = []
        inv_scale = 1.0 / scale
        for det in detections:
            aligned = align_face(image, det.landmarks * inv_scale)
            embedding = self.embed_aligned(aligned)
            results.append(
                FaceExtraction(
                    bbox=det.bbox * inv_scale,
                    landmarks=det.landmarks * inv_scale,
                    embedding=embedding,
                )
            )
        return results

    def extract_from_path(self, path: str | Path) -> list[FaceExtraction]:
        image = load_image(str(path))
        return self.extract(image)
