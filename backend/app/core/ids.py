import hashlib
import hmac
import secrets
import time
import uuid


def _namespace(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"mergenvision:{name}")


PERSON_NAMESPACE = _namespace("person")
FACE_IDENTITY_NAMESPACE = _namespace("face_identity")
PERSON_PHOTO_NAMESPACE = _namespace("person_photo")
FACE_SAMPLE_NAMESPACE = _namespace("face_sample")
PROCESS_RECORD_NAMESPACE = _namespace("process_record")


def identity_hmac(identity_key: str, master_key: str) -> str:
    """HMAC-SHA256 of the stable identity key (e.g. normalized LFW folder)."""
    return hmac.new(
        master_key.encode("utf-8"),
        identity_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def derive_person_id(identity_hmac: str) -> uuid.UUID:
    return uuid.uuid5(PERSON_NAMESPACE, identity_hmac)


def derive_face_identity_id(identity_hmac: str) -> uuid.UUID:
    return uuid.uuid5(FACE_IDENTITY_NAMESPACE, identity_hmac)


def derive_photo_id(content_sha256: str) -> uuid.UUID:
    return uuid.uuid5(PERSON_PHOTO_NAMESPACE, content_sha256)


def derive_sample_id(photo_id: uuid.UUID, model_version: str) -> uuid.UUID:
    return uuid.uuid5(FACE_SAMPLE_NAMESPACE, f"{photo_id}:{model_version}")


def derive_process_id(process_type: str, seed_bytes: bytes) -> uuid.UUID:
    return uuid.uuid5(PROCESS_RECORD_NAMESPACE, f"{process_type}:{seed_bytes.hex()}")


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7-like value (timestamp + random)."""
    timestamp_ms = int(time.time() * 1000)
    time_bytes = timestamp_ms.to_bytes(6, "big")
    rand = secrets.token_bytes(10)
    version = (rand[0] & 0x0F) | 0x70
    variant = (rand[2] & 0x3F) | 0x80
    return uuid.UUID(
        bytes=time_bytes
        + bytes([version, rand[1]])
        + bytes([variant])
        + rand[3:]
    )


def new_uuid7() -> uuid.UUID:
    return uuid7()
