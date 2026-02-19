import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported, is_sm120_supported

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
# SM120 (RTX 5090) does not support UE8M0 scale format (TCGEN05 feature)
# Only SM100/SM101 (data center Blackwell) supports it
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL and not is_sm120_supported()

if ENABLE_JIT_DEEPGEMM and is_sm120_supported():
    logger.info("SM120 detected: DeepGEMM using SM89 MMA atom (mma.sync.m16n8k32) for FP8 kernels")
