# k2-horizon-vllm

A **native [vLLM](https://github.com/vllm-project/vllm) model implementation for IFM's [K2-Horizon-MoVA](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)** — the sparse frontier model with **MoVA (Mixture-of-Values) attention** — as an out-of-tree plugin. It serves the **GPTQ Int4** quantization **on a single 48 GB GPU** (e.g. L40S / A6000) with PagedAttention, continuous batching, and Marlin kernels.

> This is a community project, not affiliated with MBZUAI IFM or the vLLM team.

## Why this exists

vLLM has no native `k2_horizon` architecture, and its **Transformers backend fails on MoVA**: the model's custom value-expert routing (`combine_routed_experts`, a Python loop over `self_attn.v_experts`) collides with vLLM's Linear substitution and its automatic MoE fusion. This plugin provides a **hand-written `K2HorizonForCausalLM`** built entirely from vLLM primitives, so the whole thing runs on the fast path.

## What's implemented

- **MoVA attention** — the value projection is a router over `mova_num_experts` value experts (top-k, SiLU), instead of a plain `v_proj`. Served as a **sparse 4-bit Marlin MoE** (reads only the top-k experts).
- **Grouped RMSNorm** (`layernorm_num_groups`), **softplus attention gate**, **sigmoid-router FFN MoE** (DeepSeek-V3 style: bias for selection only, renormalize, routed-scaling) with a shared expert, and **dense layers** for `mlp_only_layers`.
- **GPTQ Int4** weights throughout (Marlin), single-GPU.
- Works for the whole K2-Horizon family (the dense siblings like 3.7B load via the same class — useful as **speculative-decoding drafts**, they share the 250624-token vocabulary).

## The sparse value-expert kernel (the interesting bit)

The MoVA value experts are **single-projection** (`silu(W_e · x)`, hidden→kv_dim) — they don't fit vLLM's `FusedMoE` (which needs gate/up/down) and torch's fp8 grouped-GEMM isn't available on Ada. So the experts are loaded fused into one `MergedColumnParallelLinear`, and at first forward its **Marlin weights are reshaped into per-expert MoE layout** (`[K//16, E·N·2] → [E, K//16, N·2]`, expert columns are contiguous because `kv_dim` is a multiple of the Marlin N-tile) and fed to `moe_wna16_marlin_gemm` with `moe_align_block_size`. This reuses vLLM's tested 4-bit MoE kernel for a topology it wasn't built for, and reads only the top-k experts per token.

## BF16 MoVA (`K2_MOVA_BF16=1`) — for checkpoints that leave MoVA unquantized

The sparse value-expert path above reads `self.v_experts_fused.qweight`, so it requires the
MoVA value experts to be 4-bit. Not every quantized checkpoint does that — **IFM's own FP8
release deliberately does not**: its `ignored_layers` leave all 64 `self_attn.v_experts.*`
per layer *and* `self_attn.v_router` in full precision, quantizing only `mlp.experts.*`.
Such a checkpoint fails to load with
`AttributeError: 'MergedColumnParallelLinear' object has no attribute 'qweight'`.

Set `K2_MOVA_BF16=1` to build `v_experts_fused` unquantized and take a dense BF16 path:
compute all experts, SiLU, then gather the top-k and combine with the same router weights.
That is mathematically identical to the sparse path (SiLU is elementwise per expert).

It costs throughput — all `mova_num_experts` are evaluated instead of `top_k` — but the
projection is small (`E*kv_dim = 65536`, `hidden = 2560` on the 36B): ~0.13 GB of activations
per 1000 tokens. Measured on one CMP 170HX (GA100, sm_80), single-stream decode 72.8 -> 42.5
tok/s, while aggregate still reaches 874.7 tok/s at 32 concurrent.

This matters for correctness, not just compatibility: a GPTQ Int4 of this model that *does*
quantize MoVA measured **HumanEval 15.2%** on our hardware, emitting `'))))'` where `'))'` was
correct (an INT4 `v_router` selects slightly wrong value experts). The same model quantized
the way IFM does it, served through this path, measures **92.1%**.

## 4-bit KV cache on Ada (no Blackwell needed)

NVFP4 KV cache in vLLM is Blackwell-only (trtllm-gen). But vLLM also has **`int4_per_token_head`** KV — a Triton kernel that runs on **Ada (sm_89)** — which roughly doubles context vs fp8. Combined with the 4-bit value experts this reaches **~314K context on a single 48 GB card**.

## Install

```bash
pip install vllm==0.28.0 gptqmodel>=7.3.6   # torch 2.13+cu130, transformers 5.16
git clone https://github.com/stefanskiasan/k2-horizon-vllm && pip install -e k2-horizon-vllm
```
The `vllm.general_plugins` entry point registers `K2HorizonForCausalLM` automatically.

## Serve (single 48 GB GPU, max context)

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN          # required for int4 KV
vllm serve Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype int4_per_token_head \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --served-model-name k2-horizon
```
For maximum single-stream throughput at the cost of context, use `VLLM_ATTENTION_BACKEND=FLASH_ATTN --kv-cache-dtype fp8` instead.

## Reasoning & tool calling (OpenAI-compatible)

K2-Horizon emits reasoning inside `<ifm|think>…</ifm|think>` and tool calls in a nested `<ifm|tool_call>…` structure (every marker is a single token id). vLLM's stock parsers don't match this, so the plugin ships two — registered through the same `vllm.general_plugins` entry point so they load in **every** process, the API server *and* the engine core (registering only via `--tool-parser-plugin` misses the engine core):

- **`IFMReasoningParser`** (`--reasoning-parser ifm`) — surfaces the think channel as `reasoning`, streaming and non-streaming. Handles all three reasoning-effort tiers (`<ifm|think>` / `<ifm|think_fast>` / `<ifm|think_faster>`), selected per request via `chat_template_kwargs={"reasoning_effort": "high"|"medium"|"low"}`.
- **`IFMToolParser`** (`--tool-call-parser ifm`) — parses both the XML tool-call format and the `{"name":…,"arguments":…}` JSON variant into standard `tool_calls`.

```bash
vllm serve <K2-GPTQ-Int4> --trust-remote-code \
  --kv-cache-dtype int4_per_token_head \
  --enable-auto-tool-choice --tool-call-parser ifm --reasoning-parser ifm
```

> vLLM 0.28 renamed the response field `reasoning_content` → `reasoning`. If you front this with an older client or a gateway that still reads `reasoning_content`, mirror both fields (a one-line `@computed_field` on `ChatMessage`/`DeltaMessage` that returns `reasoning`).

### Tuning

Throughput is sensitive to the chunked-prefill size. On the L40S the optimum is `--max-num-batched-tokens 2048` (a broad 1024–4096 plateau; **512 is ~30 % slower** here — the "512 is best" rule some models follow does *not* hold, so measure per model/HW). `--enable-prefix-caching` helps whenever requests share a system prompt (agents); the cache hit-rate tracks prefix reuse, not chunk size.

## Performance (NVIDIA L40S, 48 GB, GPTQ Int4)

| | value |
|---|---|
| Prefill | ~6,200 tok/s |
| Decode, single stream | ~69 tok/s |
| Decode, 8 / 16 / 32 concurrent | ~300 / 530 / 1020 tok/s aggregate |
| KV context (int4 KV + 4-bit value) | ~314K tokens |

(Note: a dense K2-Horizon sibling as a draft-model is measured to be *too large* to help — its forward cost is comparable to the 4B-active target. EAGLE-3, below, is the practical route to faster single-stream.)

## EAGLE-3 speculative decoding (working, lossless)

The model implements vLLM's `SupportsEagle3` interface (auxiliary hidden states at low/mid/high layers), and a **1-layer EAGLE-3 draft head trained against it** ([vLLM Speculators](https://docs.vllm.ai/projects/speculators/)) gives a lossless single-stream speedup. A trained draft is published at **[Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3](https://huggingface.co/Siladrim/K2-Horizon-MoVA-36B-A4B-EAGLE3)**:

| config | decode tok/s (L40S) | vs. no-spec | mean accept length |
|---|---|---|---|
| no-spec baseline | 70.1 | 1.00× | — |
| **eagle3, `num_speculative_tokens:2`** | **~82** | **~1.17×** | 1.61 |

```bash
vllm serve <K2-GPTQ-Int4> --trust-remote-code \
  --kv-cache-dtype int4_per_token_head \
  --speculative-config '{"method":"eagle3","model":"<draft>","num_speculative_tokens":2}'
```

**The one non-obvious knob: train the draft with a *reduced* vocabulary** (`--draft-vocab-size 32768`, mapped back via `d2t`/`t2d`). A full-vocab draft head over the 250624-token vocabulary costs almost as much per token as the 4B-active target itself and makes decoding *slower*; the 32K head is what makes speculation net-positive. Acceptance is front-loaded (≈0.44/0.17), so `num_speculative_tokens:2` beats 1 and 3. Acceptance — and the speedup — is markedly higher on structured/agentic workloads than on the general-chat prompts benchmarked here. (The dense K2-Horizon siblings share the vocab but are too large to be efficient drafts; there is no shipped MTP head.)

### Why the speedup caps around 1.2× (the model is compute-bound)

Worth stating plainly: on Ada the Int4 GEMMs are already Marlin-optimal and K2's decode is **compute-bound**, not memory-bound. Each speculative step verifies K extra tokens ≈ K× the MoE compute, with no weight-bandwidth savings to amortize — so EAGLE-3 plateaus near ~1.2× (vs. the 2–3× typical of memory-bound models), and **suffix decoding (arctic-inference) is net-negative at every `num_speculative_tokens`** (measured −42 % at 1, and −60…−85 % at 8, even on highly repetitive production traffic). If you need faster single-stream on this model, the lever is more compute — tensor/expert-parallel across GPUs, or FP8 on Hopper — not more speculation.

## License & credit

Apache-2.0. Base model and architecture: **[IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)** by MBZUAI's Institute of Foundation Models. GPTQ Int4 weights: [Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4](https://huggingface.co/Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4).
