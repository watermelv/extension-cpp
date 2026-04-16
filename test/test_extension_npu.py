import importlib
import sys
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_PROJECT_ROOT = REPO_ROOT / "extension_cpp_stable"


def get_stable_extension():
    sys.path.insert(0, str(STABLE_PROJECT_ROOT))
    sys.modules.pop("extension_cpp_stable", None)
    importlib.invalidate_caches()
    return importlib.import_module("extension_cpp_stable")


@unittest.skipIf(not hasattr(torch, "npu") or not torch.npu.is_available(), "requires npu")
class TestStableNpuAotiShim(TestCase):
    def setUp(self):
        super().setUp()
        self.ext = get_stable_extension()
        if not hasattr(self.ext, "npu_ops") or not self.ext.npu_ops.has_npu_test_ops():
            self.skipTest("extension_cpp_stable NPU test ops were not built")

    def test_npu_raw_stream_matches_python_stream(self):
        device_index = 0
        x = torch.randn(8, device=f"npu:{device_index}", dtype=torch.float32)

        with torch.npu.device(device_index):
            expected_default = torch.npu.current_stream().npu_stream
            actual_default = self.ext.npu_ops.get_npu_raw_stream(x)
            self.assertEqual(actual_default, expected_default)

            custom_stream = torch.npu.Stream()
            with torch.npu.stream(custom_stream):
                actual_custom = self.ext.npu_ops.get_npu_raw_stream(x)
                self.assertEqual(actual_custom, custom_stream.npu_stream)

    def test_zero_size_tensor_from_blob_uses_null_data_path_on_npu(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        result = self.ext.npu_ops.make_zero_size_blob_tensor(x)

        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.layout, torch.strided)
        self.assertEqual(tuple(result.shape), (0,))
        self.assertEqual(tuple(result.stride()), (1,))
        self.assertEqual(result.numel(), 0)

    def test_zero_size_tensor_from_blob_uses_cpu_device_branch(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        result = self.ext.npu_ops.make_zero_size_cpu_blob_tensor(x)

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.layout, torch.strided)
        self.assertEqual(tuple(result.shape), (0,))
        self.assertEqual(tuple(result.stride()), (1,))
        self.assertEqual(result.numel(), 0)

    def test_cpu_non_null_blob_path_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(
            self.ext.npu_ops.check_cpu_non_null_blob_path_returns_failure(x), 1
        )

    def test_null_delete_paths_accept_nullptr_inputs(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_null_delete_paths(x), 1)

    def test_invalid_stream_guard_path_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_invalid_stream_guard_path(x), 1)

    def test_null_stream_guard_path_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_null_stream_guard_path(x), 1)

    def test_invalid_device_guard_creation_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_invalid_device_guard_creation(x), 1)

    def test_invalid_device_guard_set_index_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_invalid_device_guard_set_index(x), 1)

    def test_invalid_device_current_stream_returns_failure_internally(self):
        x = torch.randn(4, device="npu:0", dtype=torch.float32)

        self.assertEqual(self.ext.npu_ops.check_invalid_device_current_stream(x), 1)

    def test_run_npu_shim_checks_restores_device_and_stream(self):
        device_count = torch.npu.device_count()
        target_device = 1 if device_count > 1 else 0
        original_device = 0 if target_device != 0 else target_device

        torch.npu.set_device(original_device)
        x = torch.randn(16, device=f"npu:{target_device}", dtype=torch.float32)
        expected = x + 1

        with torch.npu.device(target_device):
            custom_stream = torch.npu.Stream()
            with torch.npu.stream(custom_stream):
                before_stream = torch.npu.current_stream().npu_stream
                result = self.ext.npu_ops.run_npu_shim_checks(x)
                after_stream = torch.npu.current_stream().npu_stream

        torch.testing.assert_close(result, expected)
        self.assertEqual(before_stream, custom_stream.npu_stream)
        self.assertEqual(after_stream, custom_stream.npu_stream)
        self.assertEqual(torch.npu.current_device(), original_device)


if __name__ == "__main__":
    run_tests()
