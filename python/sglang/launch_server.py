"""Launch the inference server."""

import os
import sys

# Monkey-patch missing hooks on Torch-Inductor's CompiledKernel to avoid AttributeError
try:
    from torch._inductor.runtime import triton_heuristics as _th
    if not hasattr(_th.CompiledKernel, "launch_enter_hook"):
        setattr(_th.CompiledKernel, "launch_enter_hook", lambda *args, **kwargs: None)
    if not hasattr(_th.CompiledKernel, "launch_exit_hook"):
        setattr(_th.CompiledKernel, "launch_exit_hook", lambda *args, **kwargs: None)
except ImportError:
    # Torch or its internals layout may differ; ignore if unavailable
    pass

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
