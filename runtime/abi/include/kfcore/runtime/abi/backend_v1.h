#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KFCORE_BACKEND_ABI_V1_MAJOR UINT32_C(1)
#define KFCORE_BACKEND_ABI_V1_MINOR UINT32_C(2)
#define KFCORE_BACKEND_QUERY_V1_SYMBOL "kfcore_backend_query_v1"
#define KFCORE_TENSOR_MAX_RANK_V1 UINT32_C(16)

typedef int32_t kf_status_v1;
#define KF_STATUS_V1_OK INT32_C(0)
#define KF_STATUS_V1_INVALID_ARGUMENT INT32_C(1)
#define KF_STATUS_V1_UNSUPPORTED INT32_C(2)
#define KF_STATUS_V1_INCOMPATIBLE_ARTIFACT INT32_C(3)
#define KF_STATUS_V1_DEVICE_UNAVAILABLE INT32_C(4)
#define KF_STATUS_V1_DEPENDENCY_UNAVAILABLE INT32_C(5)
#define KF_STATUS_V1_OUT_OF_MEMORY INT32_C(6)
#define KF_STATUS_V1_RUNTIME_FAILURE INT32_C(7)
#define KF_STATUS_V1_INTERNAL_FAILURE INT32_C(8)

typedef uint32_t kf_data_type_v1;
#define KF_DATA_TYPE_V1_FLOAT32 UINT32_C(1)
#define KF_DATA_TYPE_V1_FLOAT16 UINT32_C(2)
#define KF_DATA_TYPE_V1_INT8 UINT32_C(3)
#define KF_DATA_TYPE_V1_INT32 UINT32_C(4)
#define KF_DATA_TYPE_V1_INT64 UINT32_C(5)
#define KF_DATA_TYPE_V1_UINT8 UINT32_C(6)
#define KF_DATA_TYPE_V1_BOOL UINT32_C(7)
#define KF_DATA_TYPE_V1_BFLOAT16 UINT32_C(8)

typedef uint32_t kf_memory_kind_v1;
#define KF_MEMORY_KIND_V1_HOST UINT32_C(1)
#define KF_MEMORY_KIND_V1_PINNED_HOST UINT32_C(2)
#define KF_MEMORY_KIND_V1_DEVICE UINT32_C(3)

typedef uint32_t kf_tensor_io_v1;
#define KF_TENSOR_IO_V1_INPUT UINT32_C(1)
#define KF_TENSOR_IO_V1_OUTPUT UINT32_C(2)

typedef uint32_t kf_artifact_support_v1;
#define KF_ARTIFACT_SUPPORT_V1_UNSUPPORTED UINT32_C(0)
#define KF_ARTIFACT_SUPPORT_V1_SUPPORTED UINT32_C(1)

typedef uint64_t kf_backend_capability_v1;
#define KF_BACKEND_CAP_V1_HOST_MEMORY (UINT64_C(1) << 0)
#define KF_BACKEND_CAP_V1_PINNED_MEMORY (UINT64_C(1) << 1)
#define KF_BACKEND_CAP_V1_DEVICE_MEMORY (UINT64_C(1) << 2)
#define KF_BACKEND_CAP_V1_CUDA_MEMORY_INTEROP (UINT64_C(1) << 3)
#define KF_BACKEND_CAP_V1_DYNAMIC_HOST_OUTPUT (UINT64_C(1) << 4)

typedef struct kf_backend_handle_v1_t* kf_backend_handle_v1;
typedef struct kf_model_handle_v1_t* kf_model_handle_v1;
typedef struct kf_context_handle_v1_t* kf_context_handle_v1;

typedef struct kf_string_view_v1
{
    const char* data;
    uint64_t size;
} kf_string_view_v1;

typedef struct kf_backend_create_info_v1
{
    uint32_t struct_size;
} kf_backend_create_info_v1;

typedef struct kf_backend_info_v1
{
    uint32_t struct_size;
    kf_string_view_v1 backend_id;
    kf_string_view_v1 backend_name;
    uint32_t backend_version_major;
    uint32_t backend_version_minor;
    uint32_t backend_version_patch;
    kf_backend_capability_v1 capabilities;
    /* ABI v1.1 append-only fields. Zero means unknown/not applicable. */
    uint32_t execution_runtime_major;
    uint32_t execution_runtime_minor;
    uint32_t execution_runtime_patch;
} kf_backend_info_v1;

typedef struct kf_device_info_v1
{
    uint32_t struct_size;
    kf_string_view_v1 device_id;
    kf_string_view_v1 device_name;
    kf_backend_capability_v1 capabilities;
    /* ABI v1.1 append-only fields. Zero means unknown/not applicable. */
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
} kf_device_info_v1;

typedef struct kf_artifact_desc_v1
{
    uint32_t struct_size;
    kf_string_view_v1 path_utf8;
    kf_string_view_v1 format;
    kf_string_view_v1 flavor;
} kf_artifact_desc_v1;

typedef struct kf_artifact_probe_v1
{
    uint32_t struct_size;
    kf_artifact_support_v1 support;
    kf_backend_capability_v1 capabilities;
} kf_artifact_probe_v1;

typedef struct kf_model_load_info_v1
{
    uint32_t struct_size;
    kf_artifact_desc_v1 artifact;
    kf_string_view_v1 device_id;
} kf_model_load_info_v1;

typedef struct kf_context_create_info_v1
{
    uint32_t struct_size;
} kf_context_create_info_v1;

typedef struct kf_tensor_desc_v1
{
    uint32_t struct_size;
    kf_string_view_v1 name;
    kf_data_type_v1 data_type;
    const int64_t* dimensions;
    uint32_t rank;
    kf_tensor_io_v1 io;
} kf_tensor_desc_v1;

typedef struct kf_tensor_view_v1
{
    uint32_t struct_size;
    kf_string_view_v1 name;
    kf_data_type_v1 data_type;
    const int64_t* dimensions;
    uint32_t rank;
    const void* data;
    uint64_t byte_size;
    kf_memory_kind_v1 memory_kind;
    kf_string_view_v1 device_id;
} kf_tensor_view_v1;

typedef struct kf_mutable_tensor_view_v1
{
    uint32_t struct_size;
    kf_string_view_v1 name;
    kf_data_type_v1 data_type;
    const int64_t* dimensions;
    uint32_t rank;
    void* data;
    uint64_t byte_size;
    kf_memory_kind_v1 memory_kind;
    kf_string_view_v1 device_id;
} kf_mutable_tensor_view_v1;

/* ABI v1.2: caller supplies bounded Host storage; backend writes actual shape/byte count. */
typedef struct kf_dynamic_output_v1
{
    uint32_t struct_size;
    kf_string_view_v1 name;
    kf_data_type_v1 data_type;
    void* data;
    uint64_t capacity_bytes;
    kf_memory_kind_v1 memory_kind;
    kf_string_view_v1 device_id;
    int64_t dimensions[KFCORE_TENSOR_MAX_RANK_V1];
    uint32_t rank;
    uint64_t byte_size;
} kf_dynamic_output_v1;

typedef struct kf_backend_api_v1
{
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;

    kf_status_v1 (*create_backend)(const kf_backend_create_info_v1*, kf_backend_handle_v1*);
    void (*destroy_backend)(kf_backend_handle_v1);
    kf_status_v1 (*get_backend_info)(kf_backend_handle_v1, kf_backend_info_v1*);

    kf_status_v1 (*get_device_count)(kf_backend_handle_v1, uint64_t*);
    kf_status_v1 (*get_device_info)(kf_backend_handle_v1, uint64_t, kf_device_info_v1*);

    kf_status_v1 (*probe_artifact)(kf_backend_handle_v1,
                                   const kf_artifact_desc_v1*,
                                   kf_string_view_v1,
                                   kf_artifact_probe_v1*);

    kf_status_v1 (*load_model)(kf_backend_handle_v1,
                               const kf_model_load_info_v1*,
                               kf_model_handle_v1*);
    void (*destroy_model)(kf_model_handle_v1);
    kf_status_v1 (*get_tensor_count)(kf_model_handle_v1, uint64_t*);
    kf_status_v1 (*get_tensor_info)(kf_model_handle_v1, uint64_t, kf_tensor_desc_v1*);

    kf_status_v1 (*create_context)(kf_model_handle_v1,
                                   const kf_context_create_info_v1*,
                                   kf_context_handle_v1*);
    void (*destroy_context)(kf_context_handle_v1);
    kf_status_v1 (*run)(kf_context_handle_v1,
                        const kf_tensor_view_v1*,
                        uint64_t,
                        kf_mutable_tensor_view_v1*,
                        uint64_t);

    kf_status_v1 (*format_last_error)(kf_backend_handle_v1,
                                      char*,
                                      uint64_t,
                                      uint64_t*);

    /* ABI v1.2 append-only function. Dynamic output memory is Host-only in v1.2. */
    kf_status_v1 (*run_dynamic)(kf_context_handle_v1,
                                const kf_tensor_view_v1*,
                                uint64_t,
                                kf_dynamic_output_v1*,
                                uint64_t);
} kf_backend_api_v1;

typedef kf_status_v1 (*kfcore_backend_query_v1_fn)(uint32_t requested_abi_major,
                                                   uint32_t requested_abi_minor,
                                                   kf_backend_api_v1* out_api);

#ifdef __cplusplus
}
#endif
