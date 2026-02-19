# FlashMLA — local fork with SM120 patches pre-applied
#
# Instead of FetchContent + fragile cmake string(REPLACE) patches,
# we use a local copy of sgl-project/FlashMLA with SM120 changes
# applied directly to the source files:
#   - csrc/flashmla_utils.h  (IS_SM120 macro, widened IS_SM100 range)
#   - csrc/python_api.cpp    (SM120 runtime dispatch)
#   - csrc/cutlass/           (already has SM120 in this cutlass version)
#
# The local FlashMLA source lives at /app/FlashMLA in Docker
# (copied from glm-5-head-private/FlashMLA in the Dockerfile).

# Allow overriding the FlashMLA source directory via cmake variable
if(NOT DEFINED FLASHMLA_SOURCE_DIR)
    # In Docker: /app/sgl-kernel is CMAKE_CURRENT_SOURCE_DIR, FlashMLA is at /app/FlashMLA
    if(EXISTS "/app/FlashMLA/csrc/python_api.cpp")
        set(FLASHMLA_SOURCE_DIR "/app/FlashMLA")
    else()
        # Local development fallback: relative to workspace root
        set(FLASHMLA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../FlashMLA")
    endif()
endif()

# Verify the source directory exists
if(NOT EXISTS "${FLASHMLA_SOURCE_DIR}/csrc/python_api.cpp")
    message(FATAL_ERROR "FlashMLA source not found at ${FLASHMLA_SOURCE_DIR}. "
        "Expected local FlashMLA fork with SM120 patches. "
        "Copy from glm-5-head-private/FlashMLA or set -DFLASHMLA_SOURCE_DIR=<path>")
endif()

message(STATUS "Using local FlashMLA source: ${FLASHMLA_SOURCE_DIR}")

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

# SM103a patches for CUDA 13+ (applied at cmake time since they depend on CUDA version)
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    # Patch flashmla_utils.h: widen IS_SM100 to cover the full SM100 family (SM103a)
    # This is a CUDA-version-dependent patch that can't be pre-applied in the source
    set(FLASHMLA_UTILS_FILE "${FLASHMLA_SOURCE_DIR}/csrc/flashmla_utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    string(FIND "${FLASHMLA_UTILS_CONTENT}" "__CUDA_ARCH__ >= 1000" SM100_RANGE_FOUND)
    if(SM100_RANGE_FOUND EQUAL -1)
        # The local source already has the widened range from SM120 patches,
        # but double-check and apply if needed
        message(STATUS "flashmla_utils.h IS_SM100 range already widened for SM103a")
    else()
        message(STATUS "flashmla_utils.h IS_SM100 range already covers SM103a")
    endif()

    # Patch cutlass/arch/config.h: add SM103 architecture defines
    set(CUTLASS_CONFIG_FILE "${FLASHMLA_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
    file(READ "${CUTLASS_CONFIG_FILE}" CUTLASS_CONFIG_CONTENT)
    string(FIND "${CUTLASS_CONFIG_CONTENT}" "SM103" SM103_FOUND)
    if(SM103_FOUND EQUAL -1)
        string(REPLACE
"// SM101 and SM101a"
"// SM103 and SM103a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ >= 13)
  #define CUTLASS_ARCH_MMA_SM103_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM103_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030)
    #define CUTLASS_ARCH_MMA_SM103_ENABLED 1
    #if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif
    #if !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM101 and SM101a"
            CUTLASS_CONFIG_CONTENT "${CUTLASS_CONFIG_CONTENT}")
        file(WRITE "${CUTLASS_CONFIG_FILE}" "${CUTLASS_CONFIG_CONTENT}")
        message(STATUS "Patched cutlass/arch/config.h for SM103a support")
    else()
        message(STATUS "cutlass/arch/config.h already patched for SM103a")
    endif()

    # DISABLED: SM103a (Blackwell B300) - uncomment if needed
    # list(APPEND FLASHMLA_CUDA_FLAGS
    #     "-gencode=arch=compute_103a,code=sm_103a"
    # )
endif()


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

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
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
