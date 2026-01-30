from dill.tests.test_classdef import o
from jaxformers.distributed.parallel import ParallelDims
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from huggingface_hub import snapshot_download
from jax import P
from jax.sharding import AxisType, reshard
from jaxtyping import Array, Float, Int, PyTree
from safetensors import safe_open
from transformers import AddedToken, PreTrainedTokenizerFast

from jaxformers.attention_utils import eager_dot_product_attention
from jaxformers.masking_utils import AttentionMaskInterface

from ..attention_utils import AttentionInterface


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")


AxisName = str | tuple[str, ...] | None

BATCH = ("dp_replicate", "dp_shard")
FSDP = ("dp_shard", "cp")
MODEL = ("tp",)
SEQ = ("cp",)

SHARDING_RULES = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "qkv_embed": FSDP,
    "q_heads": MODEL,
    "kv_heads": MODEL,
    "o_heads": MODEL,
    "mlp_up_embed": FSDP,
    "mlp_up_ffw": None,
    "mlp_down_ffw": FSDP,
    "mlp_down_embed": MODEL,
    "vocab_in": None,
    "vocab_out": MODEL,
}

def logical_to_physical(logical, rules):
    spec = [getattr(rules, lo) for lo in logical]
    flat_leaves = jtu.tree_leaves(spec)
    if len(flat_leaves) != len(set(flat_leaves)):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {spec}"
        )

    return P(*spec)


class Config(TypedDict):
    model_type: str = "qwen3"
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: dict[str, Any]
    rope_theta: int
    sliding_window: int | None
    tie_word_embeddings: bool
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    gradient_checkpointing: bool = False
    sharding_rules: dict[str, AxisName]
    parallel_dims: ParallelDims


@dataclass
class Model:
    config: Config
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    init_kv: Callable
    tokenizer: PreTrainedTokenizerFast


def init_kv(L, K, H, B, T):
    sharding = P(None, "data", None, "model", None)
    kv = [
        jnp.zeros((2, B, T, K, H), dtype=jnp.bfloat16, out_sharding=sharding)
        for _ in range(L)
    ]
    return kv


def apply_rope(x: jax.Array, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])  # (B, T)
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))  # (H/2, )
    inp = jnp.einsum(
        "bt,h-> bth", positions, freq, precision=jax.lax.Precision.HIGHEST
    )  # (B, T, H/2)
    x1, x2 = x[:, :, :, : H // 2], x[:, :, :, H // 2 :]  # (B, T, N, H/2)
    sin, cos = (
        jnp.sin(inp).astype(x.dtype)[:, :, None, :],
        jnp.cos(inp).astype(x.dtype)[:, :, None, :],
    )  # (B, T, 1, H/2)

    return jnp.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
    )  # (B, T, N, H)


def rms_norm(x: jax.Array, gamma, eps):
    rms = jnp.sqrt(jnp.pow(x.astype(jnp.float32), 2).mean(-1, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_layer(
    cfg: Config,
    layer_idx: int,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    kv=None,
    pos=0,
    **inputs,
):
    B, T, D = x.shape

    x_norm = rms_norm(x, w["input_layernorm"], cfg["rms_norm_eps"])

    q = jnp.einsum(
        "btd,nhd->btnh",
        x_norm,
        w["q_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model", None),
    )
    k = jnp.einsum(
        "bsd,khd->bskh",
        x_norm,
        w["k_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model", None),
    )
    v = jnp.einsum(
        "bsd,khd->bskh",
        x_norm,
        w["v_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model", None),
    )

    q = rms_norm(q, w["q_norm"], cfg["rms_norm_eps"])
    k = rms_norm(k, w["k_norm"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg["rope_theta"], pos)
    k = apply_rope(k, cfg["rope_theta"], pos)

    attn_impl = cfg.get("attn_implementation", "sdpa")

    attention_interface = AttentionInterface[attn_impl]
    attn_output = attention_interface(
        q, k, v, mask=inputs["attention_mask"]
    )  # (B, T, N, H)

    o = jnp.einsum(
        "btnh,dnh->btd",
        attn_output,
        w["o_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, None),
    )
    x += o
    x_norm = rms_norm(x, w["post_attention_layernorm"], cfg["rms_norm_eps"])

    # FFN
    act_fn = jax.nn.silu
    gate = act_fn(
        jnp.einsum(
            "btd,fd->btf",
            x_norm,
            w["gate_proj"],
            preferred_element_type=jnp.float32,
            out_sharding=P("data", None, "model"),
        )
    )
    up = jnp.einsum(
        "btd,fd->btf",
        x_norm,
        w["up_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model"),
    )

    x += jnp.einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, None),
    )

    return x, kv


def forward(
    cfg: Config,
    input_ids: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    kv: None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    **inputs,
):
    B, T = input_ids.shape
    input_ids = reshard(input_ids, P("data", None))
    input_ids = (
        weights["embed_tokens"]
        .at[input_ids, :]
        .get(out_sharding=P("data", None, None))
        .astype(dtype)
    )  # (B, T, D) # think more about this mixed precision training

    return_kv = kv is not None
    if kv is None:
        kv = defaultdict(lambda: None)

    for layer_idx in range(cfg["num_hidden_layers"]):
        layer_weights = {
            k.replace(prefix, ""): v
            for k, v in weights.items()
            if (prefix := f"layers.{layer_idx}.") in k
        }
        if cfg.get("gradient_checkpointing", None):
            forward = jax.remat(partial(forward_layer, cfg, layer_idx))
        else:
            forward = partial(forward_layer, cfg, layer_idx)
        input_ids, kv[layer_idx] = forward(
            input_ids, layer_weights, kv[layer_idx], pos, **inputs
        )

    out_embed = (
        weights["embed_tokens"] if cfg["tie_word_embeddings"] else weights["lm_head"]
    )
    input_ids = rms_norm(input_ids, weights["norm"], cfg["rms_norm_eps"])
    logits = jnp.einsum(
        "btd,vd-> btv",
        input_ids,
        out_embed,
        out_sharding=P("data", None, "model"),
        preferred_element_type=input_ids.dtype,
    )

    return (logits, kv) if return_kv else logits


def save_safetensors(weights: PyTree[ModelWeights], path: epath):
    pass


def load(
    model_id: str,
    parallel_dims: ParallelDims,
    devices: list | None = None,
    hf_ckpt_dir="~/weights/huggingface",
    multihost=False,
    config_kwargs={},
    sharding_rules: dict[str, AxisName] = SHARDING_RULES,
    param_dtype: jnp.dtype = jnp.float32,
) -> Model:
    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id

    if not model_ckpt_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir)

    tokenizer_config_path = model_ckpt_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())

    tokenizer_file = str(model_ckpt_dir / "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        added_tokens_decoder={
            int(k): AddedToken(**v)
            for k, v in tokenizer_config["added_tokens_decoder"].items()
        },
    )

    cfg_path = model_ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg = Config(**cfg, **config_kwargs, parallel_dims = parallel_dims, sharding_rules = sharding_rules)
    L, N, K, H, D = (
        cfg["num_hidden_layers"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
        cfg["hidden_size"],
    )

    if multihost:
        jax.distributed.initialize()

    mesh = jax.make_mesh(
        tuple(parallel_dims.values()),
        tuple(parallel_dims.keys()),
        devices=devices,
    )
    jax.set_mesh(mesh)
    def get_sharding(key):
        if "q_proj" in key:
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        elif "k_proj" in key:
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        elif "v_proj" in key:
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        elif "gate_proj" in key:
            return logical_to_physical(("mlp_up_ffw", "ml_up_embed"), sharding_rules)
        elif "up_proj" in key:
            return logical_to_physical(("mlp_up_ffw", "ml_up_embed"), sharding_rules)
        elif "o_proj" in key:
            return logical_to_physical(("qkv_embed", "o_heads"), sharding_rules)
        elif "down_proj" in key:
            return logical_to_physical(("mlp_down_embed", "mlp_down_ffw"), sharding_rules)
        elif "embed_tokens" in key:
            return logical_to_physical(("vocab_in", "vocab_out"), sharding_rules)
        elif "lm_head" in key:
            return logical_to_physical(("vocab_in", "vocab_out"), sharding_rules)
        else:
            return P()

    weights = {}
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(
                    f.get_tensor(key).astype(param_dtype), get_sharding(key)
                )

    substrings = ["model.", "self_attn.", "mlp.", ".weight"]
    weights = {
        reduce(lambda k, s: k.replace(s, ""), substrings, k): v
        for k, v in weights.items()
    }

    for key in weights.keys():
        if "q_proj" in key:
            weights[key] = weights[key].reshape([N, H, D])
        if "k_proj" in key:
            weights[key] = weights[key].reshape([K, H, D])
        if "v_proj" in key:
            weights[key] = weights[key].reshape([K, H, D])
        if "o_proj" in key:
            weights[key] = weights[key].reshape([D, N, H])

    return Model(
        cfg, weights, partial(forward, cfg), partial(init_kv, L, K, H), tokenizer
    )
