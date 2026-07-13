import numpy as np
import pytest

from app.ml.preprocess import preprocess_detector, preprocess_recognizer, resize_image


def test_detector_preprocess_is_float32_contiguous_nchw():
    image = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)
    tensor, scale = preprocess_detector(image, 640)

    assert tensor.dtype == np.float32
    assert tensor.ndim == 4
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.flags["C_CONTIGUOUS"]
    # scale = input_size / max(h, w); here 640 / 120.
    assert scale > 1.0
    assert abs(scale - (640 / 120)) < 1e-6


def test_recognizer_preprocess_is_float32_contiguous_nchw():
    aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    tensor = preprocess_recognizer(aligned)

    assert tensor.dtype == np.float32
    assert tensor.ndim == 4
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.flags["C_CONTIGUOUS"]


def test_detector_preprocess_normalization_matches_scrfd_contract():
    # SCRFD contract: (pixel - 127.5) / 128.0, so pure white maps to 0.99609375.
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    tensor, _ = preprocess_detector(white, 640)

    assert abs(float(tensor[0, 0, 0, 0]) - 0.99609375) < 1e-6
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor_black, _ = preprocess_detector(black, 640)
    assert abs(float(tensor_black[0, 0, 0, 0]) - (-0.99609375)) < 1e-6


def test_recognizer_preprocess_normalization_matches_arcface_contract():
    # ArcFace glintr100 contract: (pixel - 127.5) / 127.5, so 255 maps to 1.0.
    white = np.full((112, 112, 3), 255, dtype=np.uint8)
    tensor = preprocess_recognizer(white)
    assert abs(float(tensor[0, 0, 0, 0]) - 1.0) < 1e-6

    black = np.zeros((112, 112, 3), dtype=np.uint8)
    tensor_black = preprocess_recognizer(black)
    assert abs(float(tensor_black[0, 0, 0, 0]) - (-1.0)) < 1e-6


def test_resize_image_preserves_aspect_ratio_and_pads_square():
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    padded, scale = resize_image(image, 640)

    assert padded.shape == (640, 640, 3)
    assert abs(scale - (640 / 200)) < 1e-6
