# Plan: Fast-Forward SGLang to Official Repository

## Current State
- **Submodule location**: `/mnt/nvme0/dc-disc-poc/worker/qwen-3.5-397b-head-private/sglang`
- **Current remote**: `https://github.com/ec-jt/sglang` (fork)
- **Current branch**: `glm-dcp-fp4`
- **Current commit**: `50a58a4a1bcda0cb660451d329adfbbbfba5225e`
- **Qwen 3.5 support**: Already present in current codebase

## Goal
Fast-forward to the latest official sglang (`sgl-project/sglang`) to get any newer Qwen 3.5 improvements.

## Execution Steps

### Step 1: Clone Official SGLang to Temp Directory
```bash
git clone --depth=1 https://github.com/sgl-project/sglang.git /tmp/sglang-official
```

### Step 2: Get Latest Commit Info from Official Repo
```bash
cd /tmp/sglang-official && git log -1 --oneline
```

### Step 3: Add Official Repo as Upstream Remote (in your submodule)
```bash
cd /mnt/nvme0/dc-disc-poc/worker/qwen-3.5-397b-head-private/sglang
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream
```

### Step 4: Check if Fast-Forward is Possible
```bash
# Check if your current commit is an ancestor of upstream/main
git merge-base --is-ancestor HEAD upstream/main && echo "FF possible" || echo "FF not possible - need merge/rebase"
```

### Step 5a: If Fast-Forward IS Possible
```bash
git checkout glm-dcp-fp4
git merge --ff-only upstream/main
```

### Step 5b: If Fast-Forward is NOT Possible
You have two options:

**Option A: Rebase (preserves your commits on top of upstream)**
```bash
git checkout glm-dcp-fp4
git rebase upstream/main
# Resolve any conflicts, then:
git rebase --continue
```

**Option B: Merge (creates a merge commit)**
```bash
git checkout glm-dcp-fp4
git merge upstream/main
# Resolve any conflicts, then:
git commit
```

### Step 6: Verify Qwen 3.5 Support After Update
```bash
# Check that Qwen 3.5 files still exist
ls -la python/sglang/srt/configs/qwen3_5.py
ls -la python/sglang/srt/models/qwen3_5*.py
```

### Step 7: Clean Up Temp Directory
```bash
rm -rf /tmp/sglang-official
```

## Important Notes

1. **Submodule Consideration**: Since this is a git submodule, after updating you may need to update the parent repo's submodule reference:
   ```bash
   cd /mnt/nvme0/dc-disc-poc/worker/qwen-3.5-397b-head-private
   git add sglang
   git commit -m "Update sglang submodule to latest official"
   ```

2. **Fork Divergence**: Your fork (`ec-jt/sglang`) may have custom changes on the `glm-dcp-fp4` branch that don't exist in the official repo. A fast-forward will only work if your branch is strictly behind upstream.

3. **Backup**: Consider creating a backup branch before any merge/rebase:
   ```bash
   git branch backup-glm-dcp-fp4
   ```

## Quick One-Liner to Check FF Feasibility
```bash
cd /mnt/nvme0/dc-disc-poc/worker/qwen-3.5-397b-head-private/sglang && \
git remote add upstream https://github.com/sgl-project/sglang.git 2>/dev/null || true && \
git fetch upstream && \
git merge-base --is-ancestor HEAD upstream/main && echo "✅ Fast-forward IS possible" || echo "❌ Fast-forward NOT possible - merge/rebase required"
```
