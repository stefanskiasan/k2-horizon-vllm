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

## Performance (NVIDIA L40S, 48 GB, GPTQ Int4)

| | value |
|---|---|
| Prefill | ~6,200 tok/s |
| Decode, single stream | ~69 tok/s |
| Decode, 8 / 16 / 32 concurrent | ~300 / 530 / 1020 tok/s aggregate |
| KV context (int4 KV + 4-bit value) | ~314K tokens |

Speculative decoding with a dense K2-Horizon sibling (3.7B, same vocabulary) as the draft pushes single-stream higher.

## License & credit

Apache-2.0. Base model and architecture: **[IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)** by MBZUAI's Institute of Foundation Models. GPTQ Int4 weights: [Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4](https://huggingface.co/Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4).
