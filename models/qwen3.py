import json
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar
from safetensors import safe_open

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jax.sharding import P, reshard
from jaxtyping import Array, Float, Int, PyTree
from transformers import AddedToken, PreTrainedTokenizerFast


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")


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
    gradient_checkpoint: bool = False


class Model:
    config: Config
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    tokenizer: PreTrainedTokenizerFast
    init_kv: Callable


def init_kv(L, K , H, B, T):
    sharding = P(None, "data", None, "model", None)
    kv =  [jnp.zeros((2, B, T, K, H), dtype = jnp.bfloat16, out_sharding = sharding) for _ in range(L)]
    return kv

def apply_rope(x: jax.Array, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])  # (B, T)
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32)) / H)  # (H/2, )
    inp = jnp.einsum("bt,h-> bth", positions, freq, precision = jax.lax.Precision.HIGHEST)  # (B, T, H/2)
    x1, x2 = x[:, :, :, : H // 2], x[:, :, :, H // 2 :]  # (B, T, N, H/2)
    sin, cos = (
        jnp.sin(inp, dtype = x.dtype)[:, :, None, :],
        jnp.cos(inp, dtype = x.dtype)[:, :, None, :],
    )  # (B, T, 1, H/2)

    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)  # (B, T, N, H)


def rms_norm(x: jax.Array, gamma, eps):
    rms = jnp.sqrt(jnp.pow(x, 2).mean(-1, keepdims=True) + eps)
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

    # self.attn
    q = jnp.einsum(
        "btd,nhd->btnh",
        x_norm,
        w["q_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, "model", None),
    )
    k = jnp.einsum(
        "bsd,khd->bskh",
        x_norm,
        w["k_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, "model", None),
    )
    v = jnp.einsum(
        "bsd,khd->bskh",
        x_norm,
        w["v_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, "model", None),
    )

    q = rms_norm(q, w["q_norm"], cfg["rms_norm_eps"])
    k = rms_norm(k, w["k_norm"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg["rope_theta"], pos)
    k = apply_rope(k, cfg["rope_theta"], pos)

    if kv is None:
        mask = jnp.tri(T, dtype=bool)
    else:
        raise NotImplementedError

    attention_interface = jax.nn.dot_product_attention
    attn_output = attention_interface(
        q, k, v, mask=mask, is_causal=True
    )  # (B, T, N, H)

    o = jnp.einsum(
        "btnh,dnh->btd",
        attn_output,
        w["o_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, None),
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
            preferred_element_type=x.dtype,
            out_sharding=("data", None, "model"),
        )
    )
    up = jnp.einsum(
        "btd,fd->btf",
        x_norm,
        w["up_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, "model"),
    )
    x += jnp.einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=("data", None, None),
    )

    return x, kv

def forward(
    cfg: Config,
    x: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    kv: None = None,
    pos: int = 0,
    **inputs,
):
    B, T = x.shape
    x = reshard(x, P("data", None))
    x = weights["embed_tokens"].take(x)  # (B, T, D)
    x = reshard(x, P("data", None, None))

    return_kv = kv is not None

    for layer_idx in range(cfg["num_hidden_layers"]):
        layer_weights = {
            k.replace(prefix, ""): v
            for k, v in weights.items()
            if (prefix := f"layers.{layer_idx}") in k
        }
        if cfg["gradient_checkpoint"]:
            forward = jax.remat(partial(forward_layer, cfg, layer_idx))
        else:
            forward = partial(forward_layer, cfg, layer_idx)
        x, kv[layer_idx] = forward(x, layer_weights, kv, pos)

    out_embed = (
        weights["embed_tokens"] if cfg["tie_word_embeddings"] else weights["lm_head"]
    )
    x = rms_norm(x, weights["norm"], cfg["rms_norm_eps"])
    logits = jnp.einsum(
        "btd,vd-> btv",
        x,
        out_embed,
        out_sharding=("data", None, "model"),
        preferred_element_type=x.dtype,
    )
    logits = jnp.einsum(
        "btd,dv-> btv",
        x,
        out_embed,
        out_sharding=("data", None, "model"),
        preferred_element_type=x.dtype,
    )
    return (logits, kv) if return_kv else logits


def load(
    model_id="Qwen/Qwen3-0.6B-Base",
    tp_devices=1,
    hf_ckpt_dir="~/weights/huggingface",
    multihost = False,
    config_kwargs = {},
    sharding_plan: Callable | None = None
):
    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id

    if not model_ckpt_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir)

    tokenizer_config_path = model_ckpt_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())

    tokenizer_file = str(model_ckpt_dir / "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, added_tokens_decoder = {int(k): AddedToken(**v) for k, v in tokenizer_config["added_tokens_decoder"]})


    cfg_path = model_ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg = Config(**cfg, **config_kwargs)
    L, N, K, H, D = cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['hidden_size']

    if multihost:
        jax.distributed.initialize()

    fsdp_devices = jax.devices() // tp_devices

    mesh = jax.make_mesh((fsdp_devices, tp_devices), ("data", "model"))
    jax.set_mesh(mesh)

    def default_get_sharding(key):
        if any(k in key for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            return P("model", "data")
        if any (k in key for k in ("o_proj", "down_proj")):
            return P("data", "model")
        if any(k in key for k in ("embed_tokens", "lm_head")):
            return P("model", "data")
        return P()

    get_sharding = sharding_plan or default_get_sharding
    weights = {}

    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(f.get_tensor(key), get_sharding(key))

    substrings = ['model.', 'self_attn.', 'mlp.', '.weight']
    weights = {reduce(lambda k, s: k.replace(s, ''), substrings, k ): v for k,v in weights.items() }


    for key in weights.key():
        if "q_proj" in key: weights[key] = weights[key].reshape([N, H, D])
        if "k_proj" in key: weights[key] = weights[key].reshape([K, H, D])
        if "v_proj" in key: weights[key] = weights[key].reshape([K, H, D])
        if "o_proj" in key: weights[key] = weights[key].reshape([D, N, H])

    return Model(cfg, weights, partial(forward, cfg), tokenizer, partial(init_kv, L, K, H))
