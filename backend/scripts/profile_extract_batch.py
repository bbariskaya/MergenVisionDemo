"""Profile a single 128-image batch."""
import asyncio, time, random
from pathlib import Path
from app.core.config import settings
from app.ml.gpu.face_pipeline import GpuFacePipeline

ROOT = Path('/app/lfw/lfw-deepfunneled/lfw-deepfunneled')
paths = [p for d in sorted(ROOT.iterdir()) if d.is_dir() for p in sorted(d.glob('*.jpg'))]
random.seed(0)
paths = random.sample(paths, 128)
buffers = [p.read_bytes() for p in paths]

async def main():
    pipeline = GpuFacePipeline(device_id=0)
    pipeline.warmup()
    _ = pipeline.extract_batch(buffers[:16])
    t0 = time.perf_counter()
    res = pipeline.extract_batch(buffers)
    t1 = time.perf_counter()
    print('faces', sum(1 for r in res if r is not None), 'time', t1-t0)
    pipeline.close()

asyncio.run(main())
