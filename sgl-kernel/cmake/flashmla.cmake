# FlashMLA CMake configuration
#
# This uses a pre-patched local copy of FlashMLA that combines:
# - PR #146 (yurekami:fix/sm120-support) for native SM120 support
# - sgl-project fork's extension/sm90/dense_fp8/ for FP8 decode kernel
#
# The local copy has been patched to:
# 1. Rename conflicting macros to FLASHMLA_* prefix (CHECK_CUDA -> FLASHMLA_CHECK_CUDA, etc.)
#    to avoid conflicts with sgl-kernel's utils.h
# 2. Add SM120a (GB200/RTX 5090) support in utils.h and dense_fp8_python_api.cpp
# 3. Widen IS_SM100 range for SM103a support
#
# Source: /app/flashmla-patched/ (copied by Dockerfile)

# Path to the pre-patched FlashMLA directory
# In Docker: /app/flashmla-patched (copied by Dockerfile)
# Locally: ${CMAKE_CURRENT_SOURCE_DIR}/../../../flashmla-patched
# Note: We check for flashmla_utils.h (renamed from utils.h to avoid include path conflicts)
if(EXISTS "/app/flashmla-patched/csrc/flashmla_utils.h")
    set(FLASHMLA_SOURCE_DIR "/app/flashmla-patched")
else()
    set(FLASHMLA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../flashmla-patched")
endif()

# Verify the patched directory exists
if(NOT EXISTS "${FLASHMLA_SOURCE_DIR}/csrc/flashmla_utils.h")
    message(FATAL_ERROR "FlashMLA patched source not found at ${FLASHMLA_SOURCE_DIR}. "
                        "Please ensure flashmla-patched directory exists with pre-patched sources.")
endif()

message(STATUS "Using pre-patched FlashMLA from: ${FLASHMLA_SOURCE_DIR}")

set(FLASHMLA_CUDA_FLAGS
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"

    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
)

# FlashMLA kernels support SM90a (Hopper), SM100a/SM103a (Blackwell), SM120a (GB200/RTX 5090)
# BUILD ONLY FOR SM120a to speed up build time (we only have GB200/RTX 5090 GPUs)
# To restore multi-arch support, uncomment the SM90a/SM100a/SM103a blocks below

# DISABLED: SM90a (Hopper H100/H200) - uncomment if needed
# if(${CUDA_VERSION} VERSION_GREATER 12.4)
#     list(APPEND FLASHMLA_CUDA_FLAGS
#         "-gencode=arch=compute_90a,code=sm_90a"
#     )
# endif()

# DISABLED: SM100a (Blackwell B100/B200) - uncomment if needed
# if(${CUDA_VERSION} VERSION_GREATER 12.8)
#     list(APPEND FLASHMLA_CUDA_FLAGS
#         "-gencode=arch=compute_100a,code=sm_100a"
#     )
# endif()

# SM120a (GB200 / RTX 5090) - ENABLED
if(${CUDA_VERSION} VERSION_GREATER 12.8)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_120a,code=sm_120a"
    )
endif()

# DISABLED: SM103a (Blackwell B300) - uncomment if needed (requires CUDA 13+)
# if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
#     list(APPEND FLASHMLA_CUDA_FLAGS
#         "-gencode=arch=compute_103a,code=sm_103a"
#     )
# endif()


set(FlashMLA_SOURCES
    "csrc/flashmla_extension.cc"
    ${FLASHMLA_SOURCE_DIR}/csrc/python_api.cpp
    ${FLASHMLA_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/smxx/mla_combine.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu

    ${FLASHMLA_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp
    ${FLASHMLA_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
    ${FLASHMLA_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
)

# Note: We do NOT use USE_SABI here because FlashMLA uses PyTorch headers (at::Tensor, c10::cuda::*)
# which transitively include pybind11 headers that require full Python API access.
# Using USE_SABI would add -DPy_LIMITED_API which breaks pybind11 compilation.
Python_add_library(flashmla_ops MODULE WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>)
target_include_directories(flashmla_ops PRIVATE
    ${FLASHMLA_SOURCE_DIR}/csrc
    ${FLASHMLA_SOURCE_DIR}/csrc/sm90
    ${FLASHMLA_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
    ${FLASHMLA_SOURCE_DIR}/csrc/cutlass/include
    ${FLASHMLA_SOURCE_DIR}/csrc/cutlass/tools/util/include
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
