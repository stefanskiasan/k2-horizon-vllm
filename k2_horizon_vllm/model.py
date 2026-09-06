# SPDX-License-Identifier: Apache-2.0
"""Native vLLM model implementation for IFM K2-Horizon-MoVA (model_type: k2_horizon).

Implements the novel MoVA (Mixture-of-Values) attention: the value projection is a
router over `mova_num_experts` value experts (top-k), combined with SiLU. The FFN is a
sigmoid-router MoE (DeepSeek-V3 style: bias only for selection, renormalize, scale) with
one shared expert. Layers in `mlp_only_layers` are dense (standard v_proj + dense MLP).
LayerNorms are GROUPED RMSNorm (layernorm_num_groups). Attention has a softplus output gate.
"""
import math
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.interfaces import SupportsEagle3, EagleModelMixin
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_make_empty_g_idx,
)
from vllm.scalar_type import scalar_types


class K2HorizonGroupedRMSNorm(nn.Module):
    """Grouped RMSNorm (n_groups): RMS is computed per group, matching the HF reference."""

    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        super().__init__()
        assert hidden_size % n_groups == 0
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.to(torch.float32)
        shape = xf.shape
        xf = xf.reshape(*shape[:-1], self.n_groups, -1)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        xf = xf.reshape(shape)
        return (self.weight * xf).to(dt)


class K2HorizonMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, quant_config, prefix, reduce_results=True):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, quant_config=quant_config,
            reduce_results=reduce_results, prefix=f"{prefix}.down_proj")
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        g, u = gate_up.chunk(2, dim=-1)
        x, _ = self.down_proj(self.act_fn(g) * u)
        return x


class K2HorizonMoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.routed_scaling_factor = config.router_scaling_factor
        # Router gate: plain fp32 linear; bias is the selection-only correction bias.
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=torch.float32)
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32))

        shared = None
        if config.num_shared_experts and config.num_shared_experts > 0:
            shared = K2HorizonMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                quant_config, prefix=f"{prefix}.shared_experts", reduce_results=False)
        self.shared_experts = shared

        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func=config.router_score_func,  # "sigmoid"
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation="silu",
            router_logits_dtype=torch.float32,
            shared_experts=shared,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states.to(torch.float32))
        out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return out.view(num_tokens, hidden_dim)


def _softplus_gate(gate_func, gate):
    if gate_func == "silu":
        return F.silu(gate)
    return F.softplus(gate, beta=math.log(2))  # softplus (K2 default)


class K2HorizonAttentionBase(nn.Module):
    """Shared attention plumbing: q/k proj, rope, paged attn, softplus gate, o_proj."""

    def _init_common(self, config, cache_config, quant_config, prefix):
        tp = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.gate_func = config.attention_gate_func

        self.q_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_heads * self.head_dim, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.k_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")
        if self.gate_func is not None:
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size, self.total_num_heads * self.head_dim, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.gate_proj")

        self.rotary_emb = get_rope(
            self.head_dim, max_position=config.max_position_embeddings,
            is_neox_style=True, rope_parameters=dict(config.rope_parameters))
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attn")

    def _finish(self, positions, hidden_states, q, k, v):
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.gate_func is not None:
            gate, _ = self.gate_proj(hidden_states)
            attn_output = attn_output * _softplus_gate(self.gate_func, gate)
        output, _ = self.o_proj(attn_output)
        return output


class K2HorizonDenseAttention(K2HorizonAttentionBase):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self._init_common(config, cache_config, quant_config, prefix)
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.v_proj")

    def forward(self, positions, hidden_states):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        return self._finish(positions, hidden_states, q, k, v)


class K2HorizonMoVAAttention(K2HorizonAttentionBase):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self._init_common(config, cache_config, quant_config, prefix)
        self.mova_num_experts = config.mova_num_experts
        self.mova_top_k = config.mova_num_experts_per_tok
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.kv_dim = self.total_num_kv_heads * self.head_dim
        # router: plain (unquantized), weight+bias used manually (bias only for selection)
        self.v_router = ReplicatedLinear(
            config.hidden_size, config.mova_num_experts, bias=config.moe_gate_bias,
            quant_config=None, prefix=f"{prefix}.v_router", return_bias=False)
        # All routed value experts fused into ONE quantized (4-bit Marlin) projection:
        # hidden -> [E * kv_dim]. Keeps experts 4-bit (~4 GB, not 15 GB BF16) AND computes
        # all experts in a single Marlin GEMM (cudagraph-safe, bandwidth-cheap).
        self.v_experts_fused = MergedColumnParallelLinear(
            config.hidden_size, [self.kv_dim] * config.mova_num_experts, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.v_experts_fused")
        self._vq = None  # lazy-built per-expert Marlin MoE weights (see _setup_sparse)

    def _mova_value(self, hidden_states):
        logits = F.linear(hidden_states, self.v_router.weight)  # weight-only, like reference
        if self.router_score_func == "softmax":
            scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            scores = torch.sigmoid(logits.to(torch.float32))
        sel = scores
        if self.v_router.bias is not None:
            sel = scores + self.v_router.bias.to(scores)
        selected = torch.topk(sel, self.mova_top_k, dim=-1).indices  # [T, top_k]
        weights = torch.gather(scores, -1, selected)
        if self.mova_top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        if self.router_scaling_factor is not None:
            weights = weights * self.router_scaling_factor
        weights = weights.to(hidden_states.dtype)

        # SPARSE 4-bit Marlin MoE over only the top-k value experts. Reuses the fused
        # Marlin weights (reshaped to per-expert MoE layout). Reads top-4/64 -> fast.
        if self._vq is None:
            self._setup_sparse()
        T = hidden_states.shape[0]
        E, N, K, tk = self.mova_num_experts, self.kv_dim, self.hidden_size, self.mova_top_k
        topk_ids = selected.to(torch.int32)
        for bsm in [8, 16, 32, 48, 64]:
            if T * tk / E / bsm < 0.9:
                break
        bsm = max(bsm, 16)
        sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids, bsm, E)
        ic = torch.empty(T * tk, N, device=hidden_states.device, dtype=hidden_states.dtype)
        ic = ops.moe_wna16_marlin_gemm(
            hidden_states, ic, self._vq, None, self._vs, None, None, None,
            self._vg, self._vsort, self._vws, sorted_ids, expert_ids, num_pad,
            weights, moe_block_size=bsm, top_k=tk, mul_topk_weights=False,
            b_q_type=scalar_types.uint4b8, size_m=T, size_n=N, size_k=K,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        ic = F.silu(ic).view(T, tk, N)                       # [T, top_k, kv_dim] token-major
        v = (ic * weights.unsqueeze(-1)).sum(dim=1)          # [T, kv_dim]
        return v.to(hidden_states.dtype)

    def _setup_sparse(self):
        """Reshape the fused Marlin weights ([K//16, E*N*2] / [K//g, E*N]) into per-expert
        MoE layout ([E, K//16, N*2] / [E, K//g, N]) and prep Marlin workspace + empty g_idx."""
        qw = self.v_experts_fused.qweight   # Marlin qweight of the fused [K, E*N] projection
        sc = self.v_experts_fused.scales    # Marlin scales
        E, N, K = self.mova_num_experts, self.kv_dim, self.hidden_size
        self._vq = qw.view(K // 16, E, N * 2).permute(1, 0, 2).contiguous()
        self._vs = sc.view(sc.shape[0], E, N).permute(1, 0, 2).contiguous()
        dev = qw.device
        self._vws = marlin_make_workspace_new(dev, 4)
        self._vg = marlin_make_empty_g_idx(dev)
        self._vsort = marlin_make_empty_g_idx(dev)
        # free the now-redundant fused Marlin weights (we only use the reshaped MoE copies)
        self.v_experts_fused.qweight.data = torch.empty(0, dtype=qw.dtype, device=dev)
        self.v_experts_fused.scales.data = torch.empty(0, dtype=sc.dtype, device=dev)

    def forward(self, positions, hidden_states):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v = self._mova_value(hidden_states)
        return self._finish(positions, hidden_states, q, k, v)


class K2HorizonDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        layer_idx = int(prefix.split(".")[-1])
        is_sparse = (layer_idx not in (config.mlp_only_layers or [])) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)

        if is_sparse and config.mova_num_experts > 0:
            self.self_attn = K2HorizonMoVAAttention(config, cache_config, quant_config, f"{prefix}.self_attn")
        else:
            self.self_attn = K2HorizonDenseAttention(config, cache_config, quant_config, f"{prefix}.self_attn")

        if is_sparse:
            self.mlp = K2HorizonMoE(config, quant_config, f"{prefix}.mlp")
        else:
            self.mlp = K2HorizonMLP(config.hidden_size, config.intermediate_size, quant_config, f"{prefix}.mlp")

        ng = config.layernorm_num_groups
        self.input_layernorm = K2HorizonGroupedRMSNorm(config.hidden_size, ng, config.rms_norm_eps)
        self.post_attention_layernorm = K2HorizonGroupedRMSNorm(config.hidden_size, ng, config.rms_norm_eps)

    def forward(self, positions, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class K2HorizonModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: K2HorizonDecoderLayer(config, cache_config, quant_config, prefix),
            prefix=f"{prefix}.layers")
        self.norm = K2HorizonGroupedRMSNorm(
            config.hidden_size, config.layernorm_num_groups, config.rms_norm_eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        # EAGLE-3: collect auxiliary hidden states at the configured layers.
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, None)
        for idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            hidden_states = layer(positions, hidden_states)
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, None)
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class K2HorizonForCausalLM(nn.Module, SupportsEagle3):
    supports_eagle3 = True
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "v_experts_fused": [f"v_experts.{i}" for i in range(64)],
    }

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple:
        n = self.config.num_hidden_layers
        return (2, n // 2, n - 3)  # low / mid / high, EAGLE-3 convention

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".mlp.gate.bias": ".mlp.gate.e_score_correction_bias"},
        orig_to_new_stacked={
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
            # 64 routed value experts -> shards of the fused Marlin projection
            **{f".v_experts.{i}.": (".v_experts_fused.", i) for i in range(64)},
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = K2HorizonModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
