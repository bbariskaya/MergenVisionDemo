import time, random
from pathlib import Path
import nvidia.nvimgcodec as nvimgcodec

ROOT = Path('/app/lfw/lfw-deepfunneled/lfw-deepfunneled')
paths = [p for d in sorted(ROOT.iterdir()) if d.is_dir() for p in sorted(d.glob('*.jpg'))]
random.seed(0)
paths = random.sample(paths, 128)
buffers = [p.read_bytes() for p in paths]

decoder = nvimgcodec.Decoder(device_id=0, backends=[], options=":fancy_upsampling=0")

# serial
t0 = time.perf_counter()
for b in buffers:
    decoder.decode(nvimgcodec.CodeStream(b), cuda_stream=0)
t1 = time.perf_counter()
print('serial decode ms', (t1-t0)*1000)

# batched
code_streams = [nvimgcodec.CodeStream(b) for b in buffers]
t0 = time.perf_counter()
imgs = decoder.decode(code_streams, cuda_stream=0)
t1 = time.perf_counter()
print('batch decode ms', (t1-t0)*1000, 'returned', len(imgs))
