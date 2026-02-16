# Plan: Upgrade SGLang to Latest Upstream While Preserving Kimi K2.5 PP Changes

## Context

### Repository Setup
- **Fork**: `https://github.com/ec-jt/sglang.git` (branch `k`)
- **Upstream**: `https://github.com/sgl-project/sglang.git` (branch `main`)
- **Submodule**: This repo lives as a git submodule at `worker/kimi-k2.5-head-private/sglang`
- **Build**: Docker-based build via `build.sh` → `docker-bake.hcl` → `Dockerfile`
- **Branch**: The `k` branch contains all Kimi K2.5 pipeline parallelism customizations

### Your Custom Changes (to preserve)
The key modifications for Kimi K2.5 pipeline parallelism support include:

1. **`python/sglang/srt/models/kimi_k25.py`** - The main model file with PP-aware logic:
   - `KimiK25ForConditionalGeneration.__init__()`: PP-aware vision tower/mm_projector creation (only on first PP rank, `PPMissingLayer` on others)
   - `get_image_feature()`: Guard to only run on first PP rank
   - `forward()`: CUDA graph capture bypass + `pp_proxy_tensors` support via `general_mm_embed_routine`
   - `load_weights()`: PP-aware weight loading (vision weights only on first rank)

2. **`python/sglang/srt/configs/kimi_k25.py`** - Config classes (`KimiK25Config`, `KimiK25VisionConfig`)

3. **`python/sglang/srt/multimodal/processors/kimi_k25.py`** - Multimodal processor for Kimi K2.5

4. **Various registration files** that wire up the model:
   - `python/sglang/srt/configs/__init__.py`
   - `python/sglang/srt/configs/model_config.py`
   - `python/sglang/srt/utils/hf_transformers_utils.py`
   - `python/sglang/srt/server_args.py`
   - `python/sglang/srt/multimodal/processors/base_processor.py`

5. **Potentially modified upstream files** for PP support in the core framework (scheduler, model_runner, cuda_graph_runner, etc.) - these may already be in upstream now.

## Strategy: Rebase onto Latest Upstream

The recommended approach is a **rebase** of the `k` branch onto the latest upstream `main`. This is preferred over merge because:
- It produces a cleaner history
- It makes it clear which commits are your custom changes vs upstream
- It's easier to re-rebase in the future for subsequent upgrades

### Alternative: Cherry-pick approach
If the rebase has too many conflicts, an alternative is to:
1. Start a fresh branch from latest upstream `main`
2. Cherry-pick your custom commits onto it
3. Resolve conflicts one commit at a time

## Step-by-Step Plan

### Phase 1: Preparation
1. Add upstream remote if not already present
2. Fetch latest upstream main
3. Identify the merge base (where your fork diverged from upstream)
4. Create a backup branch of current `k` branch state

### Phase 2: Identify Your Custom Commits
1. List all commits on `k` that are not in upstream `main`
2. Categorize them:
   - **New files** (kimi_k25.py, kimi_k25 config, processor) - unlikely to conflict
   - **Modified upstream files** (registrations, framework changes) - may conflict
3. Check if upstream has already incorporated any of your changes

### Phase 3: Rebase
1. Rebase `k` branch onto latest upstream `main`
2. Resolve conflicts file by file:
   - For new Kimi K2.5 files: should apply cleanly
   - For registration files: may need manual merge where upstream added new models
   - For core PP framework files: upstream likely has its own PP improvements - need careful merge
3. Test that the rebase result compiles/imports correctly

### Phase 4: Validation
1. Verify all Kimi K2.5 model files are intact
2. Verify PP support still works (model registrations, config, etc.)
3. Verify new upstream features are present
4. Run a basic smoke test if possible

### Phase 5: Update Build Infrastructure
1. Update the submodule reference in the parent repo
2. Rebuild Docker image to verify everything works end-to-end
3. Update any pinned versions in Dockerfile if needed (e.g., torch version, sgl-kernel)

## Commands Reference

```bash
# Phase 1: Setup
cd /mnt/nvme0/dc-disc-poc/worker/kimi-k2.5-head-private/sglang
git remote add upstream https://github.com/sgl-project/sglang.git  # if not exists
git fetch upstream
git checkout k
git branch k-backup  # safety backup

# Phase 2: Identify custom commits
git log --oneline upstream/main..k  # shows your custom commits
git diff --stat upstream/main..k    # shows changed files

# Phase 3: Rebase
git rebase upstream/main
# Resolve conflicts as they arise
# git add <resolved-file>
# git rebase --continue

# Phase 4: Verify
python -c "from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration; print('OK')"

# Phase 5: Push
git push origin k --force-with-lease
```

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Conflicts in registration files | High | These are simple additions - easy to resolve |
| Conflicts in core PP framework | Medium | Upstream has PP support already; compare approaches |
| API changes in upstream | Medium | Check if `general_mm_embed_routine`, `PPProxyTensors`, `PPMissingLayer` APIs changed |
| sgl-kernel compatibility | Low | Kernel is built separately in Docker; version pinned |
| Breaking changes in model_runner | Medium | Verify `forward()` signature and CUDA graph capture flow |

## Key Files to Watch During Rebase

These files are most likely to have conflicts:
- `python/sglang/srt/configs/__init__.py` - model registration
- `python/sglang/srt/configs/model_config.py` - architecture lists
- `python/sglang/srt/server_args.py` - model-specific settings
- `python/sglang/srt/utils/hf_transformers_utils.py` - config registration
- `python/sglang/srt/models/registry.py` - model registry
- `python/sglang/srt/model_executor/model_runner.py` - if PP APIs changed
- `python/sglang/srt/model_executor/cuda_graph_runner.py` - if capture flow changed
