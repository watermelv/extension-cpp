from . import _C, ops  # noqa: F401

try:
    import torch_npu  # noqa: F401
    from . import _C_npu_test, npu_ops  # noqa: F401
except (ImportError, OSError):
    pass
