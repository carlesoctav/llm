from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PyTree
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig

from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    logical_to_physical,
    Model,
)

from ..attention_utils import ATTENTION_INTERFACE
from ..dispatch.einsum import einsum
from ..distributed import (
    BATCH,
    CONTEXT,
    FSDP,
    MODEL,
    mutate_sharding_rule_parallel_dims,
    SEQ,
)
from ..distributed.parallel import ParallelDims
from ..masking_utils import make_bidirectional_mask


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")
AxisName = str | tuple[str, ...] | None


SHARDING_RULES = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "context": CONTEXT,
}


Config: TypeAlias = PreTrainedConfig


def layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array, eps: float):
    x_fp32 = x.astype(jnp.float32)
    mean = jnp.mean(x_fp32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_fp32 - mean), axis=-1, keepdims=True)
    x_hat = (x_fp32 - mean) / jnp.sqrt(var + eps)
    out = x_hat * gamma.astype(jnp.float32) + beta.astype(jnp.float32)
    return out.astype(x.dtype)


def get_activation_fn(hidden_act: str):
    if hidden_act == "gelu":
        return lambda x: jax.nn.gelu(x, approximate=False)
    if hidden_act == "gelu_new":
        return lambda x: jax.nn.gelu(x, approximate=True)
    if hidden_act == "relu":
        return jax.nn.relu
    if hidden_act == "silu":
        return jax.nn.silu
    raise ValueError(f"Unsupported hidden_act: {hidden_act}")


def build_bidirectional_mask(
    cfg: Config,
    input_embeds: Float[Array, "B T D"],
    attention_mask: Int[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
) -> Array | None:
    attn_impl = cfg.additional_config["attn_implementation"]
    if attn_impl not in ATTENTION_INTERFACE:
        return None

    if attention_mask is not None:
        attention_mask = attention_mask.astype(jnp.bool_)

    mask = make_bidirectional_mask(
        mask_impl=attn_impl,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    if mask is not None and attn_impl == "sdpa" and mask.ndim == 3:
        # jax.nn.dot_product_attention expects 4D mask for batched multi-head.
        return mask[:, None, :, :]
    return mask


def embed_input(
    cfg: Config,
    input_ids: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    token_type_ids: Int[Array, "B T"] | None = None,
    position_ids: Int[Array, "B T"] | None = None,
    dtype: jnp.dtype = jnp.float32,
):
    B, T = input_ids.shape
    rules = cfg.sharding_rules

    input_ids = input_ids.astype(jnp.int32)
    if token_type_ids is None:
        token_type_ids = jnp.zeros((B, T), dtype=jnp.int32)
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))

    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    token_type_ids = reshard(
        token_type_ids.astype(jnp.int32),
        logical_to_physical(("batch", "context"), rules),
    )
    position_ids = reshard(
        position_ids.astype(jnp.int32),
        logical_to_physical(("batch", "context"), rules),
    )

    word = (
        weights["bert.embeddings.word_embeddings.weight"]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
    )
    tok = (
        weights["bert.embeddings.token_type_embeddings.weight"]
        .at[token_type_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
    )
    pos = (
        weights["bert.embeddings.position_embeddings.weight"]
        .at[position_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
    )

    x = (word + tok + pos).astype(dtype)
    x = layer_norm(
        x,
        weights["bert.embeddings.LayerNorm.weight"],
        weights["bert.embeddings.LayerNorm.bias"],
        cfg.layer_norm_eps,
    )
    return x


def forward_layer(
    cfg: Config,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    layer_idx: int,
    **inputs,
):
    rules = cfg.sharding_rules
    dtype = x.dtype
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads

    q = (
        einsum(
            "btd,md->btm",
            x,
            w["q_proj_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
        + w["q_proj_b"][None, None, :]
    )
    k = (
        einsum(
            "btd,md->btm",
            x,
            w["k_proj_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
        + w["k_proj_b"][None, None, :]
    )
    v = (
        einsum(
            "btd,md->btm",
            x,
            w["v_proj_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
        + w["v_proj_b"][None, None, :]
    )

    q = rearrange(q, "b t (n h) -> b t n h", n=num_heads, h=head_dim)
    k = rearrange(k, "b t (n h) -> b t n h", n=num_heads, h=head_dim)
    v = rearrange(v, "b t (n h) -> b t n h", n=num_heads, h=head_dim)

    attn_impl = cfg.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]
    attn_output = attention_interface(q, k, v, mask=inputs.get("attn_mask"))
    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")

    attn_output = (
        einsum(
            "btd,ed->bte",
            attn_output,
            w["attn_out_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
        )
        + w["attn_out_b"][None, None, :]
    )

    # Attention residual + LN
    x = layer_norm(
        x + attn_output,
        w["attn_ln_w"],
        w["attn_ln_b"],
        cfg.layer_norm_eps,
    )
    x = reshard(x, logical_to_physical(("batch", "context", "none"), rules))

    activation = get_activation_fn(cfg.hidden_act)
    inter = activation(
        einsum(
            "btd,fd->btf",
            x,
            w["ffn_in_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
        + w["ffn_in_b"][None, None, :]
    )
    out = (
        einsum(
            "btf,df->btd",
            inter,
            w["ffn_out_w"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
        )
        + w["ffn_out_b"][None, None, :]
    )

    # FFN residual + LN
    x = layer_norm(
        x + out,
        w["ffn_ln_w"],
        w["ffn_ln_b"],
        cfg.layer_norm_eps,
    )
    return x


def forward(
    cfg: Config,
    input_ids: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    token_type_ids: Int[Array, "B T"] | None = None,
    position_ids: Int[Array, "B T"] | None = None,
    attention_mask: Int[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    return_pooled: bool = False,
    **inputs,
):
    x = embed_input(
        cfg,
        input_ids,
        weights,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        dtype=dtype,
    )

    inputs["attn_mask"] = build_bidirectional_mask(
        cfg,
        x,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"bert.encoder.layer.{layer_idx}."
        layer_weights = {
            "q_proj_w": weights[f"{prefix}attention.self.query.weight"],
            "q_proj_b": weights[f"{prefix}attention.self.query.bias"],
            "k_proj_w": weights[f"{prefix}attention.self.key.weight"],
            "k_proj_b": weights[f"{prefix}attention.self.key.bias"],
            "v_proj_w": weights[f"{prefix}attention.self.value.weight"],
            "v_proj_b": weights[f"{prefix}attention.self.value.bias"],
            "attn_out_w": weights[f"{prefix}attention.output.dense.weight"],
            "attn_out_b": weights[f"{prefix}attention.output.dense.bias"],
            "attn_ln_w": weights[f"{prefix}attention.output.LayerNorm.weight"],
            "attn_ln_b": weights[f"{prefix}attention.output.LayerNorm.bias"],
            "ffn_in_w": weights[f"{prefix}intermediate.dense.weight"],
            "ffn_in_b": weights[f"{prefix}intermediate.dense.bias"],
            "ffn_out_w": weights[f"{prefix}output.dense.weight"],
            "ffn_out_b": weights[f"{prefix}output.dense.bias"],
            "ffn_ln_w": weights[f"{prefix}output.LayerNorm.weight"],
            "ffn_ln_b": weights[f"{prefix}output.LayerNorm.bias"],
        }

        if cfg.additional_config["remat_layer"]:
            fwd = jax.remat(partial(forward_layer, cfg))
        else:
            fwd = partial(forward_layer, cfg)
        x = fwd(x, layer_weights, layer_idx, **inputs)

    if not return_pooled:
        return x

    if "bert.pooler.dense.weight" in weights and "bert.pooler.dense.bias" in weights:
        cls_state = x[:, 0, :]
        pooled = (
            einsum(
                "bd,ed->be",
                cls_state,
                weights["bert.pooler.dense.weight"],
                preferred_element_type=cls_state.dtype,
            )
            + weights["bert.pooler.dense.bias"][None, :]
        )
        pooled = jnp.tanh(pooled)
    else:
        pooled = x[:, 0, :]

    return x, pooled


def save_safetensors(weights: PyTree[ModelWeights], path: str | Path):
    pass


def load(
    model_id: str,
    parallel_dims: ParallelDims,
    devices: list | None = None,
    local_dir: str | None = None,
    multihost: bool = False,
    additional_config: AdditionalConfig | None = None,
    param_dtype: jnp.dtype = jnp.float32,
) -> Model:
    additional_config = {
        **DEFAULT_ADDITIONAL_CONFIG,
        **(additional_config or {}),
    }

    sharding_rules = mutate_sharding_rule_parallel_dims(
        dict(SHARDING_RULES),
        parallel_dims,
        sequence_parallelism=additional_config["sequence_parallelism"],
    )

    model_ckpt_dir = Path(snapshot_download(repo_id=model_id, local_dir=local_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_dir, use_fast=True)

    cfg = AutoConfig.from_pretrained(model_ckpt_dir)
    if not isinstance(cfg, PreTrainedConfig):
        raise TypeError(f"Expected HF config, got {type(cfg)!r}")
    cfg.additional_config = additional_config
    cfg.parallel_dims = parallel_dims
    cfg.sharding_rules = sharding_rules

    if multihost:
        jax.distributed.initialize()

    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    mesh = jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
    )
    jax.set_mesh(mesh)

    def canonicalize_key(key: str) -> str:
        # HF BERT checkpoints come in two common layouts:
        # - BertModel: embeddings.*, encoder.*, pooler.*
        # - BertFor*: bert.embeddings.*, bert.encoder.*, bert.pooler.*, plus task heads
        if key.startswith("bert."):
            return key
        root = key.split(".", 1)[0]
        if root in {"embeddings", "encoder", "pooler"}:
            return f"bert.{key}"
        return key

    def normalize_key(key: str) -> str:
        # Some HF BERT safetensors use TF-style LayerNorm parameter names.
        # PyTorch/HF modules expect `weight`/`bias`.
        key = key.replace(".LayerNorm.gamma", ".LayerNorm.weight")
        key = key.replace(".LayerNorm.beta", ".LayerNorm.bias")
        return key

    def get_sharding(key: str):
        try:
            if key.endswith("attention.self.query.weight"):
                return logical_to_physical(("model", "fsdp"), sharding_rules)
            if key.endswith("attention.self.key.weight"):
                return logical_to_physical(("model", "fsdp"), sharding_rules)
            if key.endswith("attention.self.value.weight"):
                return logical_to_physical(("model", "fsdp"), sharding_rules)
            if key.endswith("attention.output.dense.weight"):
                return logical_to_physical(("fsdp", "model"), sharding_rules)
            if key.endswith("intermediate.dense.weight"):
                return logical_to_physical(("model", "fsdp"), sharding_rules)
            if key.endswith("output.dense.weight"):
                return logical_to_physical(("fsdp", "model"), sharding_rules)
            if key.endswith("embeddings.word_embeddings.weight"):
                return logical_to_physical(("model", "none"), sharding_rules)
            return P()
        except Exception as e:
            print(
                f"Failed to weight shard key {key}, see {e}, defaulting to replicated"
            )
            if key.endswith("embeddings.word_embeddings.weight"):
                return logical_to_physical(("model", "none"), sharding_rules)
            return P()
        except Exception as e:
            print(f"Failed to weight shard key {key}, see {e}, defaulting to replicated")
            raise e

    weights = defaultdict(lambda: None)
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                canonical_key = canonicalize_key(key)
                normalized_key = normalize_key(canonical_key)
                arr = jax.device_put(
                    f.get_tensor(key).astype(param_dtype),
                    get_sharding(normalized_key),
                )
                weights[normalized_key] = arr
                weights[canonical_key] = arr
                if canonical_key != key:
                    weights[key] = arr

    return Model(
        config=cfg,
        weights=weights,
        forward=partial(forward, cfg),
        tokenizer=tokenizer,
    )
