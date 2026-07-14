#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

extern "C" int mergenvision_l2_normalize(
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    float epsilon,
    int* d_status,
    cudaStream_t stream);

extern "C" int mergenvision_similarity_transform(
    const float* d_landmarks,
    float* d_matrices,
    int n,
    int size,
    int* d_status,
    cudaStream_t stream);

extern "C" int mergenvision_nms(
    const float* d_boxes,
    const int* d_order,
    int n,
    float threshold,
    uint8_t* d_keep,
    cudaStream_t stream);

extern "C" int mergenvision_scale_clip_compact(
    const float* d_boxes,
    const float* d_landmarks,
    const float* d_scores,
    const int* d_order,
    const uint8_t* d_keep,
    int n,
    float inv_scale,
    int img_w,
    int img_h,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_count,
    cudaStream_t stream);

extern "C" int mergenvision_scale_clip_compact_xy(
    const float* d_boxes,
    const float* d_landmarks,
    const float* d_scores,
    const int* d_order,
    const uint8_t* d_keep,
    int n,
    float scale_x,
    float scale_y,
    int img_w,
    int img_h,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_count,
    cudaStream_t stream);

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
    cudaStream_t stream);

extern "C" int mergenvision_retinaface_decode_batch(
    const float* d_loc,
    const float* d_conf,
    const float* d_landms,
    const float* d_priors,
    int batch,
    int num_anchors,
    float conf_threshold,
    float variance0,
    float variance1,
    int max_candidates,
    float* d_out_boxes,
    float* d_out_scores,
    float* d_out_landmarks,
    int* d_counters,
    cudaStream_t stream);

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
    cudaStream_t stream);

extern "C" int mergenvision_argsort_descending(
    float* d_scores,  // modified in-place
    int* d_order,
    int n,
    cudaStream_t stream);

extern "C" int mergenvision_warp_align(
    const uint8_t* d_src,
    int h,
    int w,
    const float* d_matrices,
    int n,
    const float* d_dst,
    cudaStream_t stream);

extern "C" int mergenvision_spin_wait_cycles(
    unsigned long long cycles,
    cudaStream_t stream);


static cudaStream_t int_to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

static void check(int err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " failed: " + std::to_string(err));
    }
}

PYBIND11_MODULE(_mergenvision_gpu, m) {
    m.doc() = "MergenVision focused CUDA operators";

    m.def("l2_normalize", [](uintptr_t input_ptr, uintptr_t output_ptr,
                             int rows, int cols, float epsilon,
                             uintptr_t status_ptr, uintptr_t stream_ptr) {
        check(mergenvision_l2_normalize(
            reinterpret_cast<const float*>(input_ptr),
            reinterpret_cast<float*>(output_ptr),
            rows, cols, epsilon,
            reinterpret_cast<int*>(status_ptr),
            int_to_stream(stream_ptr)), "l2_normalize");
    }, py::arg("input_ptr"), py::arg("output_ptr"), py::arg("rows"), py::arg("cols"),
       py::arg("epsilon") = 1e-12f, py::arg("status_ptr"), py::arg("stream_ptr") = 0);

    m.def("similarity_transform", [](uintptr_t landmarks_ptr, uintptr_t matrices_ptr,
                                     int n, int size,
                                     uintptr_t status_ptr, uintptr_t stream_ptr) {
        check(mergenvision_similarity_transform(
            reinterpret_cast<const float*>(landmarks_ptr),
            reinterpret_cast<float*>(matrices_ptr),
            n, size,
            reinterpret_cast<int*>(status_ptr),
            int_to_stream(stream_ptr)), "similarity_transform");
    }, py::arg("landmarks_ptr"), py::arg("matrices_ptr"), py::arg("n"), py::arg("size"),
       py::arg("status_ptr"), py::arg("stream_ptr") = 0);

    m.def("nms", [](uintptr_t boxes_ptr, uintptr_t order_ptr,
                    int n, float threshold,
                    uintptr_t keep_ptr, uintptr_t stream_ptr) {
        check(mergenvision_nms(
            reinterpret_cast<const float*>(boxes_ptr),
            reinterpret_cast<const int*>(order_ptr),
            n, threshold,
            reinterpret_cast<uint8_t*>(keep_ptr),
            int_to_stream(stream_ptr)), "nms");
    }, py::arg("boxes_ptr"), py::arg("order_ptr"), py::arg("n"), py::arg("threshold"),
       py::arg("keep_ptr"), py::arg("stream_ptr") = 0);

    m.def("scale_clip_compact", [](uintptr_t boxes_ptr, uintptr_t landmarks_ptr,
                                   uintptr_t scores_ptr, uintptr_t order_ptr,
                                   uintptr_t keep_ptr, int n, float inv_scale,
                                   int img_w, int img_h,
                                   uintptr_t out_boxes_ptr, uintptr_t out_landmarks_ptr,
                                   uintptr_t out_scores_ptr, uintptr_t out_count_ptr,
                                   uintptr_t stream_ptr) {
        check(mergenvision_scale_clip_compact(
            reinterpret_cast<const float*>(boxes_ptr),
            reinterpret_cast<const float*>(landmarks_ptr),
            reinterpret_cast<const float*>(scores_ptr),
            reinterpret_cast<const int*>(order_ptr),
            reinterpret_cast<const uint8_t*>(keep_ptr),
            n, inv_scale, img_w, img_h,
            reinterpret_cast<float*>(out_boxes_ptr),
            reinterpret_cast<float*>(out_landmarks_ptr),
            reinterpret_cast<float*>(out_scores_ptr),
            reinterpret_cast<int*>(out_count_ptr),
            int_to_stream(stream_ptr)), "scale_clip_compact");
    }, py::arg("boxes_ptr"), py::arg("landmarks_ptr"), py::arg("scores_ptr"),
       py::arg("order_ptr"), py::arg("keep_ptr"), py::arg("n"), py::arg("inv_scale"),
       py::arg("img_w"), py::arg("img_h"),
       py::arg("out_boxes_ptr"), py::arg("out_landmarks_ptr"), py::arg("out_scores_ptr"),
       py::arg("out_count_ptr"), py::arg("stream_ptr") = 0);

    m.def("scale_clip_compact_xy", [](uintptr_t boxes_ptr, uintptr_t landmarks_ptr,
                                      uintptr_t scores_ptr, uintptr_t order_ptr,
                                      uintptr_t keep_ptr, int n,
                                      float scale_x, float scale_y,
                                      int img_w, int img_h,
                                      uintptr_t out_boxes_ptr, uintptr_t out_landmarks_ptr,
                                      uintptr_t out_scores_ptr, uintptr_t out_count_ptr,
                                      uintptr_t stream_ptr) {
        check(mergenvision_scale_clip_compact_xy(
            reinterpret_cast<const float*>(boxes_ptr),
            reinterpret_cast<const float*>(landmarks_ptr),
            reinterpret_cast<const float*>(scores_ptr),
            reinterpret_cast<const int*>(order_ptr),
            reinterpret_cast<const uint8_t*>(keep_ptr),
            n, scale_x, scale_y, img_w, img_h,
            reinterpret_cast<float*>(out_boxes_ptr),
            reinterpret_cast<float*>(out_landmarks_ptr),
            reinterpret_cast<float*>(out_scores_ptr),
            reinterpret_cast<int*>(out_count_ptr),
            int_to_stream(stream_ptr)), "scale_clip_compact_xy");
    }, py::arg("boxes_ptr"), py::arg("landmarks_ptr"), py::arg("scores_ptr"),
       py::arg("order_ptr"), py::arg("keep_ptr"), py::arg("n"),
       py::arg("scale_x"), py::arg("scale_y"),
       py::arg("img_w"), py::arg("img_h"),
       py::arg("out_boxes_ptr"), py::arg("out_landmarks_ptr"), py::arg("out_scores_ptr"),
       py::arg("out_count_ptr"), py::arg("stream_ptr") = 0);

    m.def("scrfd_decode_level", [](uintptr_t scores_ptr, uintptr_t bboxes_ptr,
                                   uintptr_t kps_ptr, uintptr_t anchors_ptr,
                                   int num_anchors, int stride, float conf_threshold,
                                   uintptr_t out_boxes_ptr, uintptr_t out_scores_ptr,
                                   uintptr_t out_landmarks_ptr, uintptr_t counter_ptr,
                                   int max_candidates, uintptr_t stream_ptr) {
        check(mergenvision_scrfd_decode_level(
            reinterpret_cast<const float*>(scores_ptr),
            reinterpret_cast<const float*>(bboxes_ptr),
            reinterpret_cast<const float*>(kps_ptr),
            reinterpret_cast<const float2*>(anchors_ptr),
            num_anchors, stride, conf_threshold,
            reinterpret_cast<float*>(out_boxes_ptr),
            reinterpret_cast<float*>(out_scores_ptr),
            reinterpret_cast<float*>(out_landmarks_ptr),
            reinterpret_cast<int*>(counter_ptr),
            max_candidates,
            int_to_stream(stream_ptr)), "scrfd_decode_level");
    }, py::arg("scores_ptr"), py::arg("bboxes_ptr"), py::arg("kps_ptr"), py::arg("anchors_ptr"),
       py::arg("num_anchors"), py::arg("stride"), py::arg("conf_threshold"),
       py::arg("out_boxes_ptr"), py::arg("out_scores_ptr"), py::arg("out_landmarks_ptr"),
       py::arg("counter_ptr"), py::arg("max_candidates"), py::arg("stream_ptr") = 0);

    m.def("retinaface_decode_batch", [](uintptr_t loc_ptr, uintptr_t conf_ptr,
                                        uintptr_t landms_ptr, uintptr_t priors_ptr,
                                        int batch, int num_anchors,
                                        float conf_threshold,
                                        float variance0, float variance1,
                                        int max_candidates,
                                        uintptr_t out_boxes_ptr, uintptr_t out_scores_ptr,
                                        uintptr_t out_landmarks_ptr, uintptr_t counters_ptr,
                                        uintptr_t stream_ptr) {
        check(mergenvision_retinaface_decode_batch(
            reinterpret_cast<const float*>(loc_ptr),
            reinterpret_cast<const float*>(conf_ptr),
            reinterpret_cast<const float*>(landms_ptr),
            reinterpret_cast<const float*>(priors_ptr),
            batch, num_anchors, conf_threshold,
            variance0, variance1, max_candidates,
            reinterpret_cast<float*>(out_boxes_ptr),
            reinterpret_cast<float*>(out_scores_ptr),
            reinterpret_cast<float*>(out_landmarks_ptr),
            reinterpret_cast<int*>(counters_ptr),
            int_to_stream(stream_ptr)), "retinaface_decode_batch");
    }, py::arg("loc_ptr"), py::arg("conf_ptr"), py::arg("landms_ptr"), py::arg("priors_ptr"),
       py::arg("batch"), py::arg("num_anchors"), py::arg("conf_threshold"),
       py::arg("variance0") = 0.1f, py::arg("variance1") = 0.2f,
       py::arg("max_candidates"),
       py::arg("out_boxes_ptr"), py::arg("out_scores_ptr"), py::arg("out_landmarks_ptr"),
       py::arg("counters_ptr"), py::arg("stream_ptr") = 0);

    m.def("retinaface_pick_largest", [](
            uintptr_t boxes_ptrs_ptr, uintptr_t landmarks_ptrs_ptr, uintptr_t scores_ptrs_ptr,
            uintptr_t counts_ptr, int n,
            uintptr_t out_boxes_ptr, uintptr_t out_landmarks_ptr,
            uintptr_t out_scores_ptr, uintptr_t out_valid_ptr,
            uintptr_t stream_ptr) {
        check(mergenvision_retinaface_pick_largest(
            reinterpret_cast<const void**>(boxes_ptrs_ptr),
            reinterpret_cast<const void**>(landmarks_ptrs_ptr),
            reinterpret_cast<const void**>(scores_ptrs_ptr),
            reinterpret_cast<const int*>(counts_ptr),
            n,
            reinterpret_cast<float*>(out_boxes_ptr),
            reinterpret_cast<float*>(out_landmarks_ptr),
            reinterpret_cast<float*>(out_scores_ptr),
            reinterpret_cast<int*>(out_valid_ptr),
            int_to_stream(stream_ptr)), "retinaface_pick_largest");
    }, py::arg("boxes_ptrs_ptr"), py::arg("landmarks_ptrs_ptr"), py::arg("scores_ptrs_ptr"),
       py::arg("counts_ptr"), py::arg("n"),
       py::arg("out_boxes_ptr"), py::arg("out_landmarks_ptr"), py::arg("out_scores_ptr"),
       py::arg("out_valid_ptr"), py::arg("stream_ptr") = 0);

    m.def("argsort_descending", [](uintptr_t scores_ptr, uintptr_t order_ptr,
                                   int n, uintptr_t stream_ptr) {
        check(mergenvision_argsort_descending(
            reinterpret_cast<float*>(scores_ptr),
            reinterpret_cast<int*>(order_ptr),
            n,
            int_to_stream(stream_ptr)), "argsort_descending");
    }, py::arg("scores_ptr"), py::arg("order_ptr"), py::arg("n"), py::arg("stream_ptr") = 0);

    m.def("warp_align", [](uintptr_t src_ptr, int h, int w,
                           uintptr_t matrices_ptr, int n,
                           uintptr_t dst_ptr, uintptr_t stream_ptr) {
        check(mergenvision_warp_align(
            reinterpret_cast<const uint8_t*>(src_ptr),
            h,
            w,
            reinterpret_cast<const float*>(matrices_ptr),
            n,
            reinterpret_cast<float*>(dst_ptr),
            int_to_stream(stream_ptr)), "warp_align");
    }, py::arg("src_ptr"), py::arg("h"), py::arg("w"),
       py::arg("matrices_ptr"), py::arg("n"),
       py::arg("dst_ptr"), py::arg("stream_ptr") = 0);

    m.def("spin_wait_cycles", [](unsigned long long cycles, uintptr_t stream_ptr) {
        check(mergenvision_spin_wait_cycles(
            cycles, int_to_stream(stream_ptr)), "spin_wait_cycles");
    }, py::arg("cycles"), py::arg("stream_ptr") = 0);
}
