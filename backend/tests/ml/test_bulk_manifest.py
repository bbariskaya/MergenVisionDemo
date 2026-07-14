import pytest

from app.services.bulk_manifest import (
    build_lfw_manifest,
    expected_cardinality,
    normalize_lfw_folder_name,
    shard_by_person_id,
)


def test_normalize_lfw_folder_name():
    key, display = normalize_lfw_folder_name("Jennifer_Aniston")
    assert key == "Jennifer_Aniston"
    assert display == "Jennifer Aniston"


@pytest.mark.skipif(
    not __import__("pathlib").Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled").exists(),
    reason="LFW dataset not mounted",
)
def test_lfw_manifest_has_person_semantics():
    root = __import__("pathlib").Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
    identities = build_lfw_manifest(root)
    persons, photos = expected_cardinality(root)

    assert len(identities) == persons
    assert sum(len(i.photos) for i in identities) == photos
    assert persons > 5000
    assert photos > 13000
    assert photos > persons

    jennifer = [i for i in identities if i.display_name == "Jennifer Aniston"]
    assert len(jennifer) == 1
    assert len(jennifer[0].photos) > 1
    assert jennifer[0].face_identity_id


@pytest.mark.skipif(
    not __import__("pathlib").Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled").exists(),
    reason="LFW dataset not mounted",
)
def test_manifest_sharding_stable():
    root = __import__("pathlib").Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
    identities = build_lfw_manifest(root)
    shards_a = shard_by_person_id(identities, 3)
    shards_b = shard_by_person_id(identities, 3)
    assert shards_a == shards_b
    assert sum(len(s) for s in shards_a) == len(identities)
    assert all(len(s) > 0 for s in shards_a)
