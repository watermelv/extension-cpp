import torch
from torch import Tensor

__all__ = [
    "has_npu_test_ops",
    "get_npu_raw_stream",
    "make_zero_size_blob_tensor",
    "make_zero_size_cpu_blob_tensor",
    "check_cpu_non_null_blob_path_returns_failure",
    "check_null_delete_paths",
    "check_invalid_stream_guard_path",
    "check_null_stream_guard_path",
    "check_invalid_device_guard_creation",
    "check_invalid_device_guard_set_index",
    "check_invalid_device_current_stream",
    "run_npu_shim_checks",
]


def has_npu_test_ops() -> bool:
    namespace = getattr(torch.ops, "extension_cpp_stable_npu_test", None)
    return namespace is not None and hasattr(namespace, "get_npu_raw_stream")


def _require_npu_test_ops() -> None:
    if has_npu_test_ops():
        return
    raise RuntimeError(
        "extension_cpp_stable NPU test ops are unavailable. "
        "Reinstall with USE_NPU=1 in an environment where torch_npu is installed."
    )


def get_npu_raw_stream(a: Tensor) -> int:
    """Returns the current raw NPU stream pointer for the tensor's device."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.get_npu_raw_stream.default(a)


def make_zero_size_blob_tensor(a: Tensor) -> Tensor:
    """Exercises the nullptr data path for an empty NPU tensor."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.make_zero_size_blob_tensor.default(a)


def make_zero_size_cpu_blob_tensor(a: Tensor) -> Tensor:
    """Exercises the CPU device branch for an empty tensor."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.make_zero_size_cpu_blob_tensor.default(a)


def check_cpu_non_null_blob_path_returns_failure(a: Tensor) -> int:
    """Exercises the current failure path for CPU tensors with a non-null data pointer."""
    _require_npu_test_ops()
    return (
        torch.ops.extension_cpp_stable_npu_test
        .check_cpu_non_null_blob_path_returns_failure
        .default(a)
    )


def check_null_delete_paths(a: Tensor) -> int:
    """Exercises delete paths that accept nullptr handles or pointers."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_null_delete_paths.default(a)


def check_invalid_stream_guard_path(a: Tensor) -> int:
    """Exercises the invalid stream error path for NPU stream guards."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_invalid_stream_guard_path.default(a)


def check_null_stream_guard_path(a: Tensor) -> int:
    """Exercises the null stream error path for NPU stream guards."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_null_stream_guard_path.default(a)


def check_invalid_device_guard_creation(a: Tensor) -> int:
    """Exercises guard creation with an invalid device index."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_invalid_device_guard_creation.default(a)


def check_invalid_device_guard_set_index(a: Tensor) -> int:
    """Exercises guard set_index with an invalid device index."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_invalid_device_guard_set_index.default(a)


def check_invalid_device_current_stream(a: Tensor) -> int:
    """Exercises current-stream lookup with an invalid device index."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.check_invalid_device_current_stream.default(a)


def run_npu_shim_checks(a: Tensor) -> Tensor:
    """Runs a suite of NPU AOTI shim validations and returns a + 1 on success."""
    _require_npu_test_ops()
    return torch.ops.extension_cpp_stable_npu_test.run_npu_shim_checks.default(a)
