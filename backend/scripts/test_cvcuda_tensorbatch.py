import ctypes, numpy as np
import cvcuda
from cuda.bindings import runtime as cuda_runtime
from app.ml.gpu.device_tensor import check_cuda

h1,w1=250,250
h2,w2=200,300
img1=np.random.randint(0,255,(h1,w1,3),dtype=np.uint8)
img2=np.random.randint(0,255,(h2,w2,3),dtype=np.uint8)

def upload(arr):
    err,ptr=cuda_runtime.cudaMalloc(arr.nbytes)
    check_cuda(err,'malloc')
    err=cuda_runtime.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes, cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice)
    check_cuda(err,'memcpy')
    return ptr, arr.shape

p1,s1=upload(img1)
p2,s2=upload(img2)

class CAI:
    def __init__(self,ptr,shape,dtype):
        self.__cuda_array_interface__={
            'shape':shape,'typestr':np.dtype(dtype).str,'data':(ptr,False),'version':2
        }

t1=cvcuda.as_tensor(CAI(p1,s1,ctypes.c_uint8), cvcuda.TensorLayout.HWC)
t2=cvcuda.as_tensor(CAI(p2,s2,ctypes.c_uint8), cvcuda.TensorLayout.HWC)
print('tensors', t1.shape, t2.shape)

try:
    tb=cvcuda.TensorBatch(2)
    tb.pushback(t1)
    tb.pushback(t2)
    print('tensorbatch', len(tb))
except Exception as e:
    print('push error', e)

try:
    resized=cvcuda.resize(tb, [(640,640),(640,640)])
    print('resize list ok', [r.shape for r in resized])
except Exception as e:
    print('resize list error', e)
