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

    # Patch python_api.cpp: Add SM120 (RTX 5090) runtime support
    # SM120 does NOT support SM100 TCGEN05 or SM90 WGMMA instructions.
    # FlashMLA decode kernels cannot run on SM120 — decode is handled by trtllm backend at Python level.
    # FlashMLA prefill sparse kernel routes SM120 to SM90 path (kernel body is empty on SM120,
    # but the host-side dispatch and metadata computation still need to work).
    set(FLASHMLA_PYTHON_API_FILE "${repo-flashmla_SOURCE_DIR}/csrc/python_api.cpp")
    file(READ "${FLASHMLA_PYTHON_API_FILE}" FLASHMLA_PYTHON_API_CONTENT)

    # Patch 1: Add is_sm120() method and widen assert_is_supported() to accept SM120
    string(REPLACE
"    bool is_sm100() const {
        return major == 10;
    }

    void assert_is_supported() const {
        TORCH_CHECK(is_sm90() || is_sm100(), \"Only SM90 and SM100 are supported\");
    }"
"    bool is_sm100() const {
        return major == 10;
    }

    bool is_sm120() const {
        return major == 12;  // RTX 5090 / GB200
    }

    void assert_is_supported() const {
        TORCH_CHECK(is_sm90() || is_sm100() || is_sm120(), \"Only SM90, SM100, and SM120 are supported\");
    }"
        FLASHMLA_PYTHON_API_CONTENT "${FLASHMLA_PYTHON_API_CONTENT}")

    # Patch 2: Add SM120 handling in get_attn_impl_meta — use SM90 formula for metadata
    # SM120 uses trtllm decode backend at Python level, but get_mla_decoding_metadata
    # is still called for prefill metadata. Route SM120 to SM90 metadata formula.
    string(REPLACE
"    } else if (arch.is_sm100()) {"
"    } else if (arch.is_sm120()) {
        // SM120: FlashMLA decode kernels not supported (no TCGEN05/WGMMA).
        // Decode is handled by trtllm backend. For prefill metadata, use SM90 formula.
        if (is_sparse_attn) {
            if (is_fp8_kvcache) {
                TORCH_CHECK(h_q_.has_value());
                int h_q = h_q_.value();
                TORCH_CHECK(h_q % h_k == 0);
                int s_q = num_q_tokens_per_head_k * h_k / h_q;
                return {
                    std::max((sm_count/2) / h_k / (cutlass::ceil_div(h_q/h_k, 2*64) * s_q), 1),
                    5,
                    64
                };
            } else {
                TORCH_CHECK(false, \"Sparse BF16 MLA is not supported on SM120\");
            }
        } else {
            if (is_fp8_kvcache) {
                TORCH_CHECK(false, \"Dense FP8 MLA is not supported on SM120\");
            } else {
                return {
                    std::max(sm_count / h_k / cutlass::ceil_div(num_q_tokens_per_head_k, 64), 1),
                    5,
                    64
                };
            }
        }
    } else if (arch.is_sm100()) {"
        FLASHMLA_PYTHON_API_CONTENT "${FLASHMLA_PYTHON_API_CONTENT}")

    # Patch 3: Add SM120 handling in fwd_kvcache_mla dispatch
    # SM120 decode is handled by trtllm at Python level, so this path should not be reached.
    # But if it is, route to SM90 sparse FP8 kernel (which has empty body on SM120 — will be a no-op).
    string(REPLACE
"    } else if (arch.is_sm100()) {
        TORCH_CHECK(is_fp8 && is_sparse_attn, \"Only FP8 + Sparse attention is supported on SM100\");
        sm100::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    } else {"
"    } else if (arch.is_sm100()) {
        TORCH_CHECK(is_fp8 && is_sparse_attn, \"Only FP8 + Sparse attention is supported on SM100\");
        sm100::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    } else if (arch.is_sm120()) {
        // SM120: FlashMLA decode kernels use TCGEN05 (SM100) or WGMMA (SM90) which SM120 lacks.
        // Decode should be handled by trtllm backend at Python level.
        // Route to SM90 sparse FP8 path as fallback (kernel body is empty on SM120).
        if (is_sparse_attn && is_fp8) {
            sm90::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
        } else if (!is_fp8) {
            if (q_dtype == torch::kBFloat16) {
                sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
            } else {
                TORCH_CHECK(false, \"Unsupported dtype for MLA on SM120\");
            }
        } else {
            TORCH_CHECK(false, \"Unsupported MLA configuration on SM120\");
        }
    } else {"
        FLASHMLA_PYTHON_API_CONTENT "${FLASHMLA_PYTHON_API_CONTENT}")

    # Patch 4: Add SM120 handling in sparse_prefill_fwd dispatch
    # Route SM120 to SM90 prefill sparse kernel (kernel body is empty on SM120 — will be a no-op).
    string(REPLACE
"    bool is_sm90 = dprops->major == 9;
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm90 || is_sm100, \"Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM90 or SM100 architectures\");"
"    bool is_sm90 = dprops->major == 9;
    bool is_sm100 = dprops->major == 10;
    bool is_sm120 = dprops->major == 12;  // SM120 (RTX 5090) — route to SM90 path
    TORCH_CHECK(is_sm90 || is_sm100 || is_sm120, \"Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM90, SM100, or SM120 architectures\");"
        FLASHMLA_PYTHON_API_CONTENT "${FLASHMLA_PYTHON_API_CONTENT}")

    # Patch 5: Route SM120 to SM90 in sparse_prefill_fwd kernel dispatch
    string(REPLACE
"    if (is_sm90) {
        sm90::run_fwd_kernel(params);
    } else if (is_sm100) {"
"    if (is_sm90 || is_sm120) {
        // SM120 routes to SM90 path (kernel body is empty on SM120 — prefill uses flashmla_sparse which calls this)
        sm90::run_fwd_kernel(params);
    } else if (is_sm100) {"
        FLASHMLA_PYTHON_API_CONTENT "${FLASHMLA_PYTHON_API_CONTENT}")

    file(WRITE "${FLASHMLA_PYTHON_API_FILE}" "${FLASHMLA_PYTHON_API_CONTENT}")
    message(STATUS "Patched python_api.cpp for SM120 (RTX 5090) runtime support")
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
