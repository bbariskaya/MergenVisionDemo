cmake_minimum_required(VERSION 3.18)
project(mergenvision_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_mergenvision_gpu
    src/bindings.cpp
    src/l2_normalize.cu
    src/similarity_transform.cu
    src/nms.cu
    src/scrfd_decode.cu
    src/retinaface_decode.cu
    src/scale_clip_compact.cu
    src/scale_clip_compact_xy.cu
    src/retinaface_pick_largest.cu
    src/argsort.cu
    src/warp_align.cu
    src/test_helpers.cu
)

target_include_directories(_mergenvision_gpu PRIVATE include)
target_link_libraries(_mergenvision_gpu PRIVATE CUDA::cudart)
set_target_properties(_mergenvision_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

install(TARGETS _mergenvision_gpu DESTINATION mergenvision_gpu)
