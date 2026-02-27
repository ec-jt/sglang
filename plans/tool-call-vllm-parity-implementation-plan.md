# SGLang Tool-Use Reliability Plan (vLLM Parity Target)

## Status

Proposed implementation plan.

## Goal

Reach vLLM-level tool-call reliability in SGLang by eliminating:

- missed tool-call extraction,
- tool JSON leaked into assistant text,
- incomplete/garbled streamed arguments,
- rare decode-loop/garbage tails after valid tool emissions.

## Success Criteria (Release Gates)

1. **Correctness parity gate**
   - Tool-call pass rate on internal regression suite >= 99.9%.
   - Zero assistant-content leakage when output is valid tool-call payload.
   - Streaming argument reconstruction exactly matches non-stream parse result.

2. **Stability gate**
   - No regressions in non-tool chat outputs.
   - No increase in malformed final JSON in tool-call mode.

3. **Performance gate**
   - P50/P95 latency regression <= 2% for chat without tools.
   - P95 streaming token interval regression <= 3% for tool mode.

## Current Baseline (SGLang)

- Tool extraction and output shaping live in [`OpenAIServingChat._process_tool_calls()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1082), [`OpenAIServingChat._process_tool_call_stream()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1245), and [`OpenAIServingChat._check_for_unstreamed_tool_args()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1351).
- Parser core is centralized in [`FunctionCallParser`](sglang/python/sglang/srt/function_call/function_call_parser.py:39), with incremental API in [`FunctionCallParser.parse_stream_chunk()`](sglang/python/sglang/srt/function_call/function_call_parser.py:121).

## vLLM Behaviors to Match

Reference behaviors to emulate:

- parser registry/contract in [`ToolParser`](tmp/vllm/vllm/tool_parsers/abstract_tool_parser.py:34) and [`ToolParserManager`](tmp/vllm/vllm/tool_parsers/abstract_tool_parser.py:122),
- robust streaming extraction/reconciliation in [`OpenAIServingChat.extract_tool_call_required_streaming()`](tmp/vllm/vllm/entrypoints/openai/chat_completion/serving.py:527),
- explicit tokenizer/request sanitization paths (`tool_call` id truncation and serialization) in [`maybe_serialize_tool_calls()`](tmp/vllm/vllm/tokenizers/mistral.py:43) and [`truncate_tool_call_ids()`](tmp/vllm/vllm/tokenizers/mistral.py:82).

## Implementation Phases

### Phase 0 — Instrumentation + Golden Corpus

Add observability and deterministic fixtures before behavior changes.

### Tasks

1. Add structured counters/tags around:
   - parse attempts, parse success/fail reason,
   - streaming partials emitted,
   - unstreamed-arg reconciliation events,
   - final `finish_reason` transitions to `tool_calls`.
2. Build golden corpus:
   - valid single-tool JSON,
   - multi-tool in one response,
   - fragmented JSON across many chunks,
   - malformed prefix then recovery,
   - tool-call + trailing garbage,
   - no-tool normal assistant answer.

### Touchpoints

- [`OpenAIServingChat._process_tool_calls()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1082)
- [`OpenAIServingChat._process_tool_call_stream()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1245)
- [`FunctionCallParser.parse_stream_chunk()`](sglang/python/sglang/srt/function_call/function_call_parser.py:121)

### Exit Criteria

- Reproducible baseline report per model family (DeepSeek/Kimi/Qwen).

---

### Phase 1 — Parser Contract Hardening (vLLM-style)

Unify parser API semantics so streaming/non-stream produce equivalent canonical outputs.

### Tasks

1. Introduce canonical parser result DTO:
   - `normal_text`,
   - `calls[]` with stable `tool_index`, `name`, `arguments_json`,
   - parser-state metadata (`is_partial`, `is_complete`, `error_kind`).
2. Ensure streaming parser emits only deltas for arguments and never duplicates previously-emitted bytes.
3. Reject undefined tool names early and downgrade to normal text only when policy says so.
4. Add strict mode to fail-closed when `tool_choice=required` and parse is structurally invalid.

### Touchpoints

- [`FunctionCallParser`](sglang/python/sglang/srt/function_call/function_call_parser.py:39)
- [`FunctionCallParser.parse_non_stream()`](sglang/python/sglang/srt/function_call/function_call_parser.py:100)
- [`FunctionCallParser.parse_stream_chunk()`](sglang/python/sglang/srt/function_call/function_call_parser.py:121)

### Exit Criteria

- Canonical equivalence test: concatenated stream parse == non-stream parse for each fixture.

---

### Phase 2 — Streaming Finalization/Reconciliation Rewrite

Make end-of-stream behavior deterministic and idempotent.

### Tasks

1. Keep per-choice parser state object (not ad-hoc text ops).
2. At stream end:
   - flush remaining argument suffix if parse can be completed safely,
   - if parser incomplete and policy allows fallback, emit as assistant text;
   - otherwise emit parser error and finish cleanly.
3. Centralize `finish_reason` mapping:
   - if any valid tool call emitted and raw finish is `stop`, map to `tool_calls`.
4. Prevent mixed-mode leakage: once tool-call mode is active for a choice, suppress conflicting plain-text deltas unless explicitly allowed.

### Touchpoints

- [`OpenAIServingChat._process_tool_call_stream()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1245)
- [`OpenAIServingChat._check_for_unstreamed_tool_args()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1351)
- [`OpenAIServingChat._process_tool_calls()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1082)

### Exit Criteria

- Zero dropped trailing-arg cases in golden streaming suite.

---

### Phase 3 — Input/Output Sanitization + Safety Rails

Port vLLM-style hygiene to reduce edge-case breakage.

### Tasks

1. Add request-time tool-call sanitization:
   - max id length clamp,
   - safe JSON normalization for tool args,
   - schema presence checks for declared tools.
2. Add output-time clamps:
   - max argument bytes per call (configurable),
   - UTF-8/JSON validity checks before emission.
3. Add guard against decode loops/garbage tails in tool mode:
   - repeated-pattern detector,
   - max trailing non-JSON token budget after first valid tool block,
   - forced early stop + deterministic finalization.

### Touchpoints

- [`OpenAIServingChat._validate_request()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:192)
- [`OpenAIServingChat._convert_to_internal_request()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:240)
- [`OpenAIServingChat._generate_chat_stream()`](sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:613)

### Exit Criteria

- No tool-id/arg overflow failures in fuzz tests.
- Garbage-tail incidents reduced to zero in replay corpus.

---

### Phase 4 — Compatibility Layer + Feature Flags

Roll out safely with reversible controls.

### Tasks

1. Add staged flags:
   - `tool_parser_v2_enabled`,
   - `tool_stream_finalize_v2_enabled`,
   - `tool_mode_strict_output_enabled`.
2. Default ON in canary profiles, OFF for legacy behavior initially.
3. Add per-model parser selection/override map.

### Exit Criteria

- 1-week canary without critical regressions.

## Test Matrix

1. **Unit tests**
   - parser DTO conversion,
   - partial chunk assembly,
   - finish-reason mapping,
   - strict/lenient policy branches.

2. **Integration tests (OpenAI API)**
   - stream=true and stream=false parity,
   - multi-choice (`n>1`) isolation,
   - mixed reasoning + tool calls,
   - tool-required behavior.

3. **Fuzz tests**
   - random chunk boundaries,
   - unicode/control chars,
   - truncated JSON and nested objects.

4. **Replay tests**
   - production failure logs replayed against old/new parser path.

## Telemetry/Monitoring

Expose metrics:

- `tool_parse_attempt_total`,
- `tool_parse_success_total`,
- `tool_parse_failure_total{reason=...}`,
- `tool_stream_reconcile_total`,
- `tool_output_leakage_total`,
- `tool_finish_reason_override_total`.

Add request-level debug snapshots behind sampling to capture parser state transitions.

## Rollout Strategy

1. Land Phase 0 first and baseline metrics.
2. Enable Phase 1+2 behind flags in staging.
3. Canary by model family and traffic shard.
4. Enable sanitization/loop rails globally.
5. Flip v2 defaults once error budget is stable for 7 days.

## Rollback Plan

- Immediate: disable all v2 flags and revert to legacy parser pipeline.
- Partial: keep instrumentation + sanitization while rolling back stream finalization.
- Hard rollback criteria:
  - tool parse success drops > 0.3%,
  - p95 latency regression > 5%,
  - any severe customer-facing malformed-tool incident spike.

## Deliverables

1. Parser v2 contract + implementation.
2. Streaming finalization v2 path.
3. Sanitization and anti-garbage rails.
4. Full automated test suite and replay harness.
5. Runbook + docs update for parser flags and model-specific parser configuration.
