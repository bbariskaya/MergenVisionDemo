import secrets
import time
import uuid


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
