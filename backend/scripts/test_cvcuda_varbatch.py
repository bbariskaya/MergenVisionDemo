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

i1=cvcuda.as_image(t1.cuda())
i2=cvcuda.as_image(t2.cuda())
print('images', i1.width, i1.height, i2.width, i2.height)

ib=cvcuda.ImageBatchVarShape(2)
ib.pushback(i1)
ib.pushback(i2)
print('batch', len(ib))

try:
    resized=cvcuda.resize(ib, [(640,640),(640,640)], cvcuda.Interp.LINEAR)
    print('resized', type(resized), len(resized))
except Exception as e:
    print('resize error', e)

# means for RGB->BGR
means=np.array([[0,0,1,-104],[0,1,0,-117],[1,0,0,-123]], dtype=np.float32)
err,mptr=cuda_runtime.cudaMalloc(means.nbytes)
cuda_runtime.cudaMemcpy(mptr, means.ctypes.data, means.nbytes, cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice)
twist=cvcuda.as_tensor(CAI(mptr, means.shape, ctypes.c_float))

try:
    f32=cvcuda.convertto(resized, cvcuda.Type.F32, scale=1.0, offset=0.0)
    print('f32', type(f32))
    twisted=cvcuda.color_twist(f32, twist)
    print('twisted', type(twisted))
    # Need NCHW output. reformat may accept ImageBatchVarShape?
    nchw=cvcuda.reformat(twisted, cvcuda.TensorLayout.NCHW)
    print('nchw', type(nchw))
except Exception as e:
    print('process error', e)
