from pathlib import Path

import pytest


SAMPLES_DIR = Path("/app/artifacts/samples")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nvidia.nvimgcodec") is None,
    reason="nvidia-nvimgcodec-cu12 not installed",
)
def test_nvimgcodec_decodes_jpeg_to_gpu():
    import numpy as np
    from PIL import Image

    import nvidia.nvimgcodec as nvimgcodec

    jpeg_path = SAMPLES_DIR / "t1.jpg"
    jpeg_bytes = jpeg_path.read_bytes()

    decoder = nvimgcodec.Decoder(device_id=0)
    image = decoder.decode(nvimgcodec.CodeStream(jpeg_bytes))

    # Default decode is device-backed.
    cuda_view = image.cuda()
    cai = cuda_view.__cuda_array_interface__
    device_ptr = cai["data"][0]
    assert device_ptr != 0
    assert len(cai["shape"]) == 3
    assert cai["shape"][2] == 3  # RGB

    # Optional host comparison: ensure decoded content matches PIL expectation.
    pil = Image.open(jpeg_path).convert("RGB")
    cpu_view = image.cpu()
    decoded_arr = np.asarray(cpu_view)
    assert decoded_arr.shape[:2] == (pil.height, pil.width)
