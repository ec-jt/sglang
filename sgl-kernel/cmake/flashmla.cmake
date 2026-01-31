include(FetchContent)

# flash_mla
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/sgl-project/FlashMLA
    GIT_TAG be055fb7df0090fde45f08e9cb5b8b4c0272da73
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-flashmla)

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

    # Patch flashmla_utils.h for SM120a (Grace-Blackwell GB200) support
    set(FLASHMLA_UTILS_FILE "${repo-flashmla_SOURCE_DIR}/csrc/flashmla_utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    
    # Check if IS_SM120 macro already exists
    string(FIND "${FLASHMLA_UTILS_CONTENT}" "IS_SM120" SM120_MACRO_FOUND)
    if(SM120_MACRO_FOUND EQUAL -1)
        # Add IS_SM120 macro after IS_SM100 definition
        string(REPLACE
            "#define IS_SM100 0
#endif"
            "#define IS_SM100 0
#endif

// SM120a (Grace-Blackwell GB200) detection
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define IS_SM120 1
#else
#define IS_SM120 0
#endif"
            FLASHMLA_UTILS_CONTENT "${FLASHMLA_UTILS_CONTENT}")
        file(WRITE "${FLASHMLA_UTILS_FILE}" "${FLASHMLA_UTILS_CONTENT}")
        message(STATUS "Patched flashmla_utils.h for SM120a (GB200) support")
    else()
        message(STATUS "flashmla_utils.h already patched for SM120a")
    endif()

    # Patch cutlass/arch/config.h for SM120a architecture defines
    set(CUTLASS_CONFIG_FILE "${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
    file(READ "${CUTLASS_CONFIG_FILE}" CUTLASS_CONFIG_CONTENT)
    string(FIND "${CUTLASS_CONFIG_CONTENT}" "SM120" SM120_CUTLASS_FOUND)
    if(SM120_CUTLASS_FOUND EQUAL -1)
        # Add SM120a support block before SM100 block
        string(REPLACE
"// SM100 and SM100a"
"// SM120 and SM120a (Grace-Blackwell GB200)
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1
  #define CUTLASS_ARCH_MMA_SM120A_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM120_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200)
    #define CUTLASS_ARCH_MMA_SM120_ENABLED 1
    #define CUTLASS_ARCH_MMA_SM120A_ENABLED 1
    // SM120a inherits SM100a capabilities
    #if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif
    #if !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM100 and SM100a"
            CUTLASS_CONFIG_CONTENT "${CUTLASS_CONFIG_CONTENT}")
        file(WRITE "${CUTLASS_CONFIG_FILE}" "${CUTLASS_CONFIG_CONTENT}")
        message(STATUS "Patched cutlass/arch/config.h for SM120a (GB200) support")
    else()
        message(STATUS "cutlass/arch/config.h already patched for SM120a")
    endif()

    # Patch dense_fp8_python_api.cpp to allow SM120 in addition to SM90
    # The original code checks: dprops->major == 9 && dprops->minor == 0
    # We need to also allow SM120 (dprops->major == 12)
    set(DENSE_FP8_API_FILE "${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp")
    if(EXISTS "${DENSE_FP8_API_FILE}")
        file(READ "${DENSE_FP8_API_FILE}" DENSE_FP8_API_CONTENT)
        string(FIND "${DENSE_FP8_API_CONTENT}" "dprops->major == 12" SM120_FP8_FOUND)
        if(SM120_FP8_FOUND EQUAL -1)
            # Patch the SM90 check to also allow SM120
            # Original: dprops->major == 9 && dprops->minor == 0
            # New: (dprops->major == 9 || dprops->major == 12) && dprops->minor == 0
            string(REPLACE
                "dprops->major == 9 && dprops->minor == 0"
                "(dprops->major == 9 || dprops->major == 12) && dprops->minor == 0"
                DENSE_FP8_API_CONTENT "${DENSE_FP8_API_CONTENT}")
            file(WRITE "${DENSE_FP8_API_FILE}" "${DENSE_FP8_API_CONTENT}")
            message(STATUS "Patched dense_fp8_python_api.cpp for SM120 (GB200) support")
        else()
            message(STATUS "dense_fp8_python_api.cpp already patched for SM120")
        endif()
    else()
        message(WARNING "dense_fp8_python_api.cpp not found at ${DENSE_FP8_API_FILE}")
    endif()
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    # Patch FlashMLA sources for SM103a support.
    # These patches are only needed (and only valid) with CUDA 13+.

    # Patch flashmla_utils.h: widen IS_SM100 to cover the full SM100 family
    set(FLASHMLA_UTILS_FILE "${repo-flashmla_SOURCE_DIR}/csrc/flashmla_utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    string(REPLACE
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1"
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
#define IS_SM100 1"
        FLASHMLA_UTILS_CONTENT "${FLASHMLA_UTILS_CONTENT}")
    file(WRITE "${FLASHMLA_UTILS_FILE}" "${FLASHMLA_UTILS_CONTENT}")
    message(STATUS "Patched flashmla_utils.h for SM103a support")

    # Patch cutlass/arch/config.h: add SM103 architecture defines.
    # The new block is inserted right before the existing "// SM101 and SM101a"
    # anchor in the upstream header.
    set(CUTLASS_CONFIG_FILE "${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
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
    ${repo-flashmla_SOURCE_DIR}/csrc/python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/mla_combine.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu

    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
)

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>)
target_include_directories(flashmla_ops PRIVATE
    ${repo-flashmla_SOURCE_DIR}/csrc
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
