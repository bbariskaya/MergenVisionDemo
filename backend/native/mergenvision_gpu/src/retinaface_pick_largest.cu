#include <cuda_runtime.h>
#include <cstddef>

__global__ static void pick_largest_kernel(
    const float* const* d_boxes,
    const float* const* d_landmarks,
    const float* const* d_scores,
    const int* d_counts,
    int n,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_valid
) {
    int img = blockIdx.x;
    if (img >= n) {
        return;
    }

    int count = d_counts[img];
    const float* boxes = d_boxes[img];
    const float* landmarks = d_landmarks[img];
    const float* scores = d_scores[img];

    int best_idx = -1;
    float best_area = -1.0f;

    for (int k = 0; k < count; ++k) {
        float x1 = boxes[k * 4 + 0];
        float y1 = boxes[k * 4 + 1];
        float x2 = boxes[k * 4 + 2];
        float y2 = boxes[k * 4 + 3];
        float area = (x2 - x1) * (y2 - y1);
        if (area > best_area) {
            best_area = area;
            best_idx = k;
        }
    }

    if (best_idx >= 0) {
        for (int d = 0; d < 4; ++d) {
            d_out_boxes[img * 4 + d] = boxes[best_idx * 4 + d];
        }
        for (int d = 0; d < 10; ++d) {
            d_out_landmarks[img * 10 + d] = landmarks[best_idx * 10 + d];
        }
        d_out_scores[img] = scores[best_idx];
        d_out_valid[img] = 1;
    } else {
        d_out_valid[img] = 0;
    }
}

extern "C" int mergenvision_retinaface_pick_largest(
    const void** h_boxes_ptrs,
    const void** h_landmarks_ptrs,
    const void** h_scores_ptrs,
    const int* h_counts,
    int n,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_valid,
    cudaStream_t stream
) {
    void *d_boxes = nullptr;
    void *d_landmarks = nullptr;
    void *d_scores = nullptr;
    void *d_counts = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_boxes, n * sizeof(void*));
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_landmarks, n * sizeof(void*));
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_scores, n * sizeof(void*));
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_counts, n * sizeof(int));
    if (err != cudaSuccess) return static_cast<int>(err);

    err = cudaMemcpyAsync(d_boxes, h_boxes_ptrs, n * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpyAsync(d_landmarks, h_landmarks_ptrs, n * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpyAsync(d_scores, h_scores_ptrs, n * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpyAsync(d_counts, h_counts, n * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    pick_largest_kernel<<<n, 1, 0, stream>>>(
        reinterpret_cast<const float* const*>(d_boxes),
        reinterpret_cast<const float* const*>(d_landmarks),
        reinterpret_cast<const float* const*>(d_scores),
        reinterpret_cast<const int*>(d_counts),
        n,
        d_out_boxes,
        d_out_landmarks,
        d_out_scores,
        d_out_valid
    );

    cudaStreamSynchronize(stream);

    cudaFree(d_boxes);
    cudaFree(d_landmarks);
    cudaFree(d_scores);
    cudaFree(d_counts);

    return 0;
}
