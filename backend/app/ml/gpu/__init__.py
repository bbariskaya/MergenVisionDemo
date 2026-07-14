"""Production GPU data plane modules.

These modules must not import PIL, OpenCV image processing, NumPy image
processing, PyTorch, CuPy, or any CPU-fallback path in the hot path.
"""
