import hashlib
import hmac


def hash_national_id(raw_national_id: str, key: str) -> str:
    if not raw_national_id or not key:
        raise ValueError("raw national ID and key required")
    return hmac.new(key.encode(), raw_national_id.encode(), hashlib.sha256).hexdigest()


def mask_national_id(raw_national_id: str) -> str:
    if len(raw_national_id) <= 4:
        return "*" * len(raw_national_id)
    return "*" * (len(raw_national_id) - 4) + raw_national_id[-4:]
