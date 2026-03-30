"""
MitraSETI GPU-Accelerated Algorithms

Optional GPU backends for compute-intensive pipeline stages.
Falls back gracefully to CPU (Rust core) when no GPU is available.

Modules:
    taylor_tree_gpu  — CuPy-based Taylor tree de-Doppler search
"""
