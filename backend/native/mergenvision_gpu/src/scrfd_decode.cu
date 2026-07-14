#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

// Decode one FPN level. Each thread handles one anchor.
__global__ void decode_level_kernel(
    // inputs
    const float* scores,  // [A, 1]
    const float* bboxes,  // [A, 4]
    const float* kps,     // [A, 10]
    const float2* anchors, // [A]
    int A,
    int stride,
    float conf_threshold,
    // outputs (preallocated)
    float* out_boxes,     // [max_candidates, 4]
    float* out_scores,    // [max_candidates]
    float* out_landmarks, // [max_candidates, 10]
    int* counter,
    int max_candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= A) return;

    float score = scores[idx];
    if (score < conf_threshold) return;

    int pos = atomicAdd(counter, 1);
    if (pos >= max_candidates) return;

    float cx = anchors[idx].x;
    float cy = anchors[idx].y;

    const float* bb = bboxes + idx * 4;
    out_boxes[pos * 4 + 0] = cx - bb[0] * stride;
    out_boxes[pos * 4 + 1] = cy - bb[1] * stride;
    out_boxes[pos * 4 + 2] = cx + bb[2] * stride;
    out_boxes[pos * 4 + 3] = cy + bb[3] * stride;

    out_scores[pos] = score;

    const float* kp = kps + idx * 10;
    float* ol = out_landmarks + pos * 10;
    for (int k = 0; k < 5; ++k) {
        ol[k * 2 + 0] = cx + kp[k * 2 + 0] * stride;
        ol[k * 2 + 1] = cy + kp[k * 2 + 1] * stride;
    }
}

extern "C" int mergenvision_scrfd_decode_level(
    const float* d_scores,
    const float* d_bboxes,
    const float* d_kps,
    const float2* d_anchors,
    int num_anchors,
    int stride,
    float conf_threshold,
    float* d_out_boxes,
    float* d_out_scores,
    float* d_out_landmarks,
    int* d_counter,
    int max_candidates,
    cudaStream_t stream)
{
    constexpr int block = 256;
    int grid = (num_anchors + block - 1) / block;
    decode_level_kernel<<<grid, block, 0, stream>>>(
        d_scores, d_bboxes, d_kps, d_anchors, num_anchors, stride,
        conf_threshold, d_out_boxes, d_out_scores, d_out_landmarks,
        d_counter, max_candidates);
    return cudaGetLastError();
}

} // namespace mergenvision
