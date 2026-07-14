from pathlib import Path
import sys
sys.path.insert(0, "/app")

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.core.config import settings
from app.ml.gpu.decoder import JpegGpuDecoder
from app.ml.gpu.device_tensor import check_cuda
from app.ml.gpu.preprocess import GpuDetectorPreprocessor
from app.ml.preprocess import load_image, preprocess_detector


def main():
    jpeg_path = Path("/app/artifacts/samples/t1.jpg")
    jpeg_bytes = jpeg_path.read_bytes()

    decoder = JpegGpuDecoder(device_id=0)
    preprocessor = GpuDetectorPreprocessor(
        input_size=settings.detector_input_size,
        device_id=0,
    )

    err, stream = cuda_runtime.cudaStreamCreate()
    check_cuda(err, "stream create")
    h = int(stream)

    decoded, info = decoder.decode(jpeg_bytes, stream=h)
    print("info", info)

    gpu_input = preprocessor.preprocess(decoded, stream=h)
    print("gpu_input", gpu_input.shape, gpu_input.dtype)

    host = np.empty(gpu_input.shape, dtype=np.float32)
    cuda_runtime.cudaMemcpyAsync(
        host.ctypes.data,
        gpu_input.ptr,
        gpu_input.nbytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        h,
    )
    check_cuda(cuda_runtime.cudaStreamSynchronize(h), "sync")
    print("finite", np.isfinite(host).all())

    cpu_image = load_image(str(jpeg_path))
    cpu_tensor, _ = preprocess_detector(cpu_image, settings.detector_input_size)
    cpu_host = cpu_tensor.astype(np.float32)
    print("max_abs_err", float(np.max(np.abs(host - cpu_host))))

    # Bind the GPU-preprocessed tensor directly into the detector engine.
    from app.ml.gpu.trt_device_engine import TrtDeviceEngine
    det = TrtDeviceEngine(settings.detector_engine_path, device_id=0)
    det_outputs = det.infer_device({"input.1": gpu_input}, stream=h)
    print("detector outputs", sorted(det_outputs.keys()))
    for name, tensor in det_outputs.items():
        assert tensor.device_id == 0
        print(f"  {name}: {tensor.shape}")


if __name__ == "__main__":
    main()
