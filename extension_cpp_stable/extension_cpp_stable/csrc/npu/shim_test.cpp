#include <Python.h>

#include <ATen/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/add.h>
#include <c10/core/DeviceType.h>
#include <torch/library.h>

#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/inductor/aoti_torch/c/shim.h>
#include <torch_npu/csrc/inductor/aoti_torch/utils.h>

#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

extern "C" {
PyObject* PyInit__C_npu_test(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C_npu_test",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace {

std::mutex& stream_registry_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<void*, c10::Stream>& stream_registry()
{
    static std::unordered_map<void*, c10::Stream> registry;
    return registry;
}

void remember_stream(const c10_npu::NPUStream& stream)
{
    auto* raw_stream = reinterpret_cast<void*>(stream.stream(false));
    std::lock_guard<std::mutex> lock(stream_registry_mutex());
    stream_registry().insert_or_assign(raw_stream, static_cast<c10::Stream>(stream));
}

void check_aoti_error(AOTITorchError err, const char* call)
{
    TORCH_CHECK(err == AOTI_TORCH_SUCCESS, call, " failed with error code ", err);
}

std::vector<int64_t> to_vector(c10::IntArrayRef values)
{
    return std::vector<int64_t>(values.begin(), values.end());
}

void assert_npu_tensor(const at::Tensor& tensor, const char* op_name)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        op_name,
        " expects an NPU tensor");
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat,
        op_name,
        " currently tests float32 tensors only");
    TORCH_CHECK(
        tensor.layout() == at::kStrided,
        op_name,
        " expects a strided tensor");
}

int64_t get_npu_raw_stream_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "get_npu_raw_stream");

    void* shim_stream = nullptr;
    const auto device_index = tensor.device().index();
    check_aoti_error(
        aoti_torch_get_current_npu_stream(device_index, &shim_stream),
        "aoti_torch_get_current_npu_stream");

    auto direct_stream =
        reinterpret_cast<void*>(c10_npu::getCurrentNPUStream(device_index).stream(false));
    TORCH_CHECK(
        shim_stream == direct_stream,
        "aoti_torch_get_current_npu_stream returned ",
        shim_stream,
        ", but c10_npu returned ",
        direct_stream);
    return reinterpret_cast<int64_t>(shim_stream);
}

at::Tensor make_zero_size_blob_tensor_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "make_zero_size_blob_tensor");

    const auto device_index = tensor.device().index();
    std::vector<int64_t> sizes = {0};
    std::vector<int64_t> strides = {1};
    AtenTensorHandle handle = nullptr;
    check_aoti_error(
        aoti_torch_create_tensor_from_blob_npu(
            nullptr,
            static_cast<int64_t>(sizes.size()),
            sizes.data(),
            strides.data(),
            0,
            aoti_torch_dtype_float32(),
            aoti_torch_device_type_npu(),
            device_index,
            &handle),
        "aoti_torch_create_tensor_from_blob_npu(nullptr, npu)");

    at::Tensor result =
        *torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
    TORCH_CHECK(result.device() == tensor.device(), "Zero-size NPU tensor device mismatch");
    TORCH_CHECK(result.scalar_type() == at::kFloat, "Zero-size NPU tensor dtype mismatch");
    TORCH_CHECK(result.layout() == at::kStrided, "Zero-size NPU tensor layout mismatch");
    TORCH_CHECK(result.sizes().vec() == sizes, "Zero-size NPU tensor sizes mismatch");
    TORCH_CHECK(result.strides().vec() == strides, "Zero-size NPU tensor strides mismatch");
    TORCH_CHECK(result.numel() == 0, "Zero-size NPU tensor should be empty");
    check_aoti_error(
        aoti_torch_delete_tensor_object(handle),
        "aoti_torch_delete_tensor_object(zero_size_npu)");
    return result;
}

at::Tensor make_zero_size_cpu_blob_tensor_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "make_zero_size_cpu_blob_tensor");

    std::vector<int64_t> sizes = {0};
    std::vector<int64_t> strides = {1};
    AtenTensorHandle handle = nullptr;
    check_aoti_error(
        aoti_torch_create_tensor_from_blob_npu(
            nullptr,
            static_cast<int64_t>(sizes.size()),
            sizes.data(),
            strides.data(),
            0,
            aoti_torch_dtype_float32(),
            aoti_torch_device_type_cpu(),
            0,
            &handle),
        "aoti_torch_create_tensor_from_blob_npu(nullptr, cpu)");

    at::Tensor result =
        *torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
    TORCH_CHECK(result.device().type() == c10::DeviceType::CPU, "Zero-size CPU tensor device mismatch");
    TORCH_CHECK(result.scalar_type() == at::kFloat, "Zero-size CPU tensor dtype mismatch");
    TORCH_CHECK(result.layout() == at::kStrided, "Zero-size CPU tensor layout mismatch");
    TORCH_CHECK(result.sizes().vec() == sizes, "Zero-size CPU tensor sizes mismatch");
    TORCH_CHECK(result.strides().vec() == strides, "Zero-size CPU tensor strides mismatch");
    TORCH_CHECK(result.numel() == 0, "Zero-size CPU tensor should be empty");
    check_aoti_error(
        aoti_torch_delete_tensor_object(handle),
        "aoti_torch_delete_tensor_object(zero_size_cpu)");
    return result;
}

int64_t check_null_delete_paths_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_null_delete_paths");

    check_aoti_error(
        aoti_torch_delete_npu_guard(nullptr),
        "aoti_torch_delete_npu_guard(nullptr)");
    check_aoti_error(
        aoti_torch_delete_npu_stream_guard(nullptr),
        "aoti_torch_delete_npu_stream_guard(nullptr)");
    check_aoti_error(
        aoti_torch_npu_caching_allocator_raw_delete(nullptr),
        "aoti_torch_npu_caching_allocator_raw_delete(nullptr)");
    return 1;
}

int64_t check_invalid_stream_guard_path_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_invalid_stream_guard_path");

    NPUStreamGuardHandle guard = nullptr;
    auto err = aoti_torch_create_npu_stream_guard(
        reinterpret_cast<void*>(0x1),
        tensor.device().index(),
        &guard);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "Invalid stream pointer should make aoti_torch_create_npu_stream_guard fail");
    TORCH_CHECK(guard == nullptr, "Guard should remain nullptr when stream guard creation fails");
    return 1;
}

int64_t check_cpu_non_null_blob_path_returns_failure_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_cpu_non_null_blob_path_returns_failure");

    auto cpu_base = tensor.cpu().contiguous();
    auto sizes = to_vector(cpu_base.sizes());
    auto strides = to_vector(cpu_base.strides());
    AtenTensorHandle handle = nullptr;
    auto err = aoti_torch_create_tensor_from_blob_npu(
        cpu_base.data_ptr(),
        cpu_base.dim(),
        sizes.data(),
        strides.data(),
        cpu_base.storage_offset(),
        aoti_torch_dtype_float32(),
        aoti_torch_device_type_cpu(),
        0,
        &handle);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "CPU + non-null data path should currently fail in aoti_torch_create_tensor_from_blob_npu");
    TORCH_CHECK(
        handle == nullptr,
        "Tensor handle should remain nullptr when CPU + non-null data path fails");
    return 1;
}

int64_t check_null_stream_guard_path_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_null_stream_guard_path");

    NPUStreamGuardHandle guard = nullptr;
    auto err =
        aoti_torch_create_npu_stream_guard(nullptr, tensor.device().index(), &guard);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "Null stream pointer should make aoti_torch_create_npu_stream_guard fail");
    TORCH_CHECK(guard == nullptr, "Guard should remain nullptr when null stream guard creation fails");
    return 1;
}

int64_t check_invalid_device_guard_creation_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_invalid_device_guard_creation");

    const auto invalid_device_index = static_cast<int32_t>(c10_npu::device_count());
    NPUGuardHandle guard = nullptr;
    auto err = aoti_torch_create_npu_guard(invalid_device_index, &guard);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "Invalid device index should make aoti_torch_create_npu_guard fail");
    TORCH_CHECK(guard == nullptr, "Guard should remain nullptr when invalid guard creation fails");
    return 1;
}

int64_t check_invalid_device_guard_set_index_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_invalid_device_guard_set_index");

    const auto original_device = c10_npu::current_device();
    const auto valid_device_index = tensor.device().index();
    const auto invalid_device_index = static_cast<int32_t>(c10_npu::device_count());
    NPUGuardHandle guard = nullptr;
    check_aoti_error(
        aoti_torch_create_npu_guard(valid_device_index, &guard),
        "aoti_torch_create_npu_guard(valid for set_index)");
    TORCH_CHECK(
        c10_npu::current_device() == valid_device_index,
        "Valid guard creation should switch to the requested device");

    auto err = aoti_torch_npu_guard_set_index(guard, invalid_device_index);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "Invalid device index should make aoti_torch_npu_guard_set_index fail");
    TORCH_CHECK(
        c10_npu::current_device() == valid_device_index,
        "Failed guard set_index should keep the previously selected device");

    check_aoti_error(
        aoti_torch_delete_npu_guard(guard),
        "aoti_torch_delete_npu_guard(valid after failed set_index)");
    TORCH_CHECK(
        c10_npu::current_device() == original_device,
        "Deleting the guard should restore the original device after failed set_index");
    return 1;
}

int64_t check_invalid_device_current_stream_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "check_invalid_device_current_stream");

    const auto invalid_device_index = static_cast<int32_t>(c10_npu::device_count());
    void* stream = nullptr;
    auto err = aoti_torch_get_current_npu_stream(invalid_device_index, &stream);
    TORCH_CHECK(
        err == AOTI_TORCH_FAILURE,
        "Invalid device index should make aoti_torch_get_current_npu_stream fail");
    return 1;
}

at::Tensor run_npu_shim_checks_impl(const at::Tensor& tensor)
{
    assert_npu_tensor(tensor, "run_npu_shim_checks");

    const auto device_index = tensor.device().index();
    TORCH_CHECK(
        aoti_torch_device_type_npu() ==
            static_cast<int32_t>(c10::DeviceType::PrivateUse1),
        "aoti_torch_device_type_npu should return PrivateUse1");

    void* shim_stream = nullptr;
    check_aoti_error(
        aoti_torch_get_current_npu_stream(device_index, &shim_stream),
        "aoti_torch_get_current_npu_stream");
    auto current_stream =
        reinterpret_cast<void*>(c10_npu::getCurrentNPUStream(device_index).stream(false));
    TORCH_CHECK(
        shim_stream == current_stream,
        "Current stream mismatch: shim=",
        shim_stream,
        ", direct=",
        current_stream);

    const auto original_device = c10_npu::current_device();
    NPUGuardHandle guard = nullptr;
    check_aoti_error(
        aoti_torch_create_npu_guard(device_index, &guard),
        "aoti_torch_create_npu_guard");
    TORCH_CHECK(
        c10_npu::current_device() == device_index,
        "NPU guard did not switch the current device");
    check_aoti_error(
        aoti_torch_npu_guard_set_index(guard, device_index),
        "aoti_torch_npu_guard_set_index");
    TORCH_CHECK(
        c10_npu::current_device() == device_index,
        "NPU guard set_index did not keep the requested device");
    check_aoti_error(
        aoti_torch_delete_npu_guard(guard),
        "aoti_torch_delete_npu_guard");
    TORCH_CHECK(
        c10_npu::current_device() == original_device,
        "Deleting the NPU guard did not restore the original device");

    auto original_stream =
        reinterpret_cast<void*>(c10_npu::getCurrentNPUStream(device_index).stream(false));
    auto pooled_stream_obj = c10_npu::getNPUStreamFromPool(device_index);
    remember_stream(pooled_stream_obj);
    auto pooled_stream =
        reinterpret_cast<void*>(pooled_stream_obj.stream(false));
    NPUStreamGuardHandle stream_guard = nullptr;
    check_aoti_error(
        aoti_torch_create_npu_stream_guard(pooled_stream, device_index, &stream_guard),
        "aoti_torch_create_npu_stream_guard");
    auto guarded_stream =
        reinterpret_cast<void*>(c10_npu::getCurrentNPUStream(device_index).stream(false));
    TORCH_CHECK(
        guarded_stream == pooled_stream,
        "NPU stream guard did not switch to the pooled stream");
    check_aoti_error(
        aoti_torch_delete_npu_stream_guard(stream_guard),
        "aoti_torch_delete_npu_stream_guard");
    auto restored_stream =
        reinterpret_cast<void*>(c10_npu::getCurrentNPUStream(device_index).stream(false));
    TORCH_CHECK(
        restored_stream == original_stream,
        "Deleting the NPU stream guard did not restore the original stream");

    void* zero_alloc = reinterpret_cast<void*>(0x1);
    check_aoti_error(
        aoti_torch_npu_caching_allocator_raw_alloc(0, &zero_alloc),
        "aoti_torch_npu_caching_allocator_raw_alloc(0)");
    TORCH_CHECK(
        zero_alloc == nullptr,
        "Zero-byte NPU allocation should return nullptr");

    void* alloc_ptr = nullptr;
    check_aoti_error(
        aoti_torch_npu_caching_allocator_raw_alloc(64, &alloc_ptr),
        "aoti_torch_npu_caching_allocator_raw_alloc");
    TORCH_CHECK(alloc_ptr != nullptr, "NPU allocator returned nullptr");
    check_aoti_error(
        aoti_torch_npu_caching_allocator_raw_delete(alloc_ptr),
        "aoti_torch_npu_caching_allocator_raw_delete");

    auto contiguous = tensor.contiguous();
    auto sizes = to_vector(contiguous.sizes());
    auto strides = to_vector(contiguous.strides());

    AtenTensorHandle alias_handle = nullptr;
    check_aoti_error(
        aoti_torch_create_tensor_from_blob_npu(
            contiguous.data_ptr(),
            contiguous.dim(),
            sizes.data(),
            strides.data(),
            contiguous.storage_offset(),
            aoti_torch_dtype_float32(),
            aoti_torch_device_type_npu(),
            device_index,
            &alias_handle),
        "aoti_torch_create_tensor_from_blob_npu");
    at::Tensor alias =
        *torch::aot_inductor::tensor_handle_to_tensor_pointer(alias_handle);
    TORCH_CHECK(alias.device() == contiguous.device(), "Alias tensor device mismatch");
    TORCH_CHECK(alias.scalar_type() == contiguous.scalar_type(), "Alias tensor dtype mismatch");
    TORCH_CHECK(alias.data_ptr() == contiguous.data_ptr(), "Alias tensor data_ptr mismatch");
    TORCH_CHECK(
        to_vector(alias.sizes()) == sizes,
        "Alias tensor sizes mismatch");
    TORCH_CHECK(
        to_vector(alias.strides()) == strides,
        "Alias tensor strides mismatch");
    TORCH_CHECK(
        alias.storage_offset() == contiguous.storage_offset(),
        "Alias tensor storage_offset mismatch");
    TORCH_CHECK(alias.equal(contiguous), "Alias tensor value mismatch");
    check_aoti_error(
        aoti_torch_delete_tensor_object(alias_handle),
        "aoti_torch_delete_tensor_object(alias_handle)");

    AtenTensorHandle alias_v2_handle = nullptr;
    check_aoti_error(
        aoti_torch_create_tensor_from_blob_npu_v2(
            contiguous.data_ptr(),
            contiguous.dim(),
            sizes.data(),
            strides.data(),
            contiguous.storage_offset(),
            aoti_torch_dtype_float32(),
            aoti_torch_device_type_npu(),
            device_index,
            &alias_v2_handle,
            aoti_torch_layout_strided(),
            nullptr,
            0),
        "aoti_torch_create_tensor_from_blob_npu_v2");
    at::Tensor alias_v2 =
        *torch::aot_inductor::tensor_handle_to_tensor_pointer(alias_v2_handle);
    TORCH_CHECK(
        alias_v2.data_ptr() == contiguous.data_ptr(),
        "Alias tensor v2 data_ptr mismatch");
    TORCH_CHECK(alias_v2.equal(contiguous), "Alias tensor v2 value mismatch");

    AtenTensorHandle mkldnn_handle = nullptr;
    auto mkldnn_err = aoti_torch_create_tensor_from_blob_npu_v2(
        contiguous.data_ptr(),
        contiguous.dim(),
        sizes.data(),
        strides.data(),
        contiguous.storage_offset(),
        aoti_torch_dtype_float32(),
        aoti_torch_device_type_npu(),
        device_index,
        &mkldnn_handle,
        static_cast<int32_t>(at::kMkldnn),
        nullptr,
        0);
    TORCH_CHECK(
        mkldnn_err == AOTI_TORCH_FAILURE,
        "Mkldnn layout should be rejected on NPU");

    auto result = at::add(alias_v2, 1.0);
    check_aoti_error(
        aoti_torch_delete_tensor_object(alias_v2_handle),
        "aoti_torch_delete_tensor_object(alias_v2_handle)");
    return result;
}

} // namespace

namespace c10_npu {

NPUStream getNPUStreamFromAclrtStream(aclrtStream stream, c10::DeviceIndex device_index)
{
    TORCH_CHECK(stream != nullptr, "stream is nullptr");

    auto try_registered_stream = [stream]() -> std::optional<NPUStream> {
        std::lock_guard<std::mutex> lock(stream_registry_mutex());
        auto it = stream_registry().find(reinterpret_cast<void*>(stream));
        if (it == stream_registry().end()) {
            return std::nullopt;
        }
        return NPUStream(NPUStream::UNCHECKED, it->second);
    };
    if (auto registered = try_registered_stream()) {
        return *registered;
    }

    auto try_known_streams = [stream](c10::DeviceIndex idx) -> std::optional<NPUStream> {
        auto current = getCurrentNPUStream(idx);
        if (current.stream(false) == stream) {
            remember_stream(current);
            return current;
        }

        auto default_stream = getDefaultNPUStream(idx);
        if (default_stream.stream(false) == stream) {
            remember_stream(default_stream);
            return default_stream;
        }

        return std::nullopt;
    };

    if (device_index != -1) {
        if (auto known = try_known_streams(device_index)) {
            return *known;
        }
    } else {
        const auto device_count = c10_npu::device_count();
        for (c10::DeviceIndex idx = 0; idx < device_count; ++idx) {
            if (auto known = try_known_streams(idx)) {
                return *known;
            }
        }
    }

    TORCH_CHECK(
        false,
        "The aclrtStream is not managed by the shim test registry on device ",
        device_index);
}

} // namespace c10_npu

TORCH_LIBRARY(extension_cpp_stable_npu_test, m)
{
    m.def("get_npu_raw_stream(Tensor a) -> int");
    m.def("make_zero_size_blob_tensor(Tensor a) -> Tensor");
    m.def("make_zero_size_cpu_blob_tensor(Tensor a) -> Tensor");
    m.def("check_cpu_non_null_blob_path_returns_failure(Tensor a) -> int");
    m.def("check_null_delete_paths(Tensor a) -> int");
    m.def("check_invalid_stream_guard_path(Tensor a) -> int");
    m.def("check_null_stream_guard_path(Tensor a) -> int");
    m.def("check_invalid_device_guard_creation(Tensor a) -> int");
    m.def("check_invalid_device_guard_set_index(Tensor a) -> int");
    m.def("check_invalid_device_current_stream(Tensor a) -> int");
    m.def("run_npu_shim_checks(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp_stable_npu_test, PrivateUse1, m)
{
    m.impl("get_npu_raw_stream", &get_npu_raw_stream_impl);
    m.impl("make_zero_size_blob_tensor", &make_zero_size_blob_tensor_impl);
    m.impl("make_zero_size_cpu_blob_tensor", &make_zero_size_cpu_blob_tensor_impl);
    m.impl(
        "check_cpu_non_null_blob_path_returns_failure",
        &check_cpu_non_null_blob_path_returns_failure_impl);
    m.impl("check_null_delete_paths", &check_null_delete_paths_impl);
    m.impl("check_invalid_stream_guard_path", &check_invalid_stream_guard_path_impl);
    m.impl("check_null_stream_guard_path", &check_null_stream_guard_path_impl);
    m.impl("check_invalid_device_guard_creation", &check_invalid_device_guard_creation_impl);
    m.impl("check_invalid_device_guard_set_index", &check_invalid_device_guard_set_index_impl);
    m.impl("check_invalid_device_current_stream", &check_invalid_device_current_stream_impl);
    m.impl("run_npu_shim_checks", &run_npu_shim_checks_impl);
}
