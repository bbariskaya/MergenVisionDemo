from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket_photos: str = "mergenvision-photos"
    minio_secure: bool = False

    qdrant_url: str
    qdrant_collection: str = "face_samples"

    hmac_key: str

    worker_role: str = "online"
    vggface_dataset_path: Path = Path("/datasets/vgg-face")
    vggface_target_active_photos: int = 1_000_000
    bulk_extract_batch_size: int = 512
    bulk_max_persistence_concurrency: int = 32
    bulk_activation_batch_size: int = 2048

    model_pack: str = "antelopev2"
    inference_backend: str = "tensorrt"
    detector_model_path: Path = Path("/models/detector.onnx")
    embedder_model_path: Path = Path("/models/recognizer.onnx")
    detector_engine_path: Path = Path("/engines/detector.engine")
    embedder_engine_path: Path = Path("/engines/recognizer.engine")
    detector_input_size: int = 640
    embedder_input_size: int = 112
    embedding_dim: int = 512
    detector_confidence_threshold: float = 0.5
    detector_nms_iou: float = 0.4
    matched_threshold: float = 0.6
    possible_threshold: float = 0.4
    top_k_default: int = 5
    top_k_max: int = 20

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
