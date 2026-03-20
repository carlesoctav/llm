import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxformers.utils import GeneralInterface


BlockMask = Array
MaskImpl = tp.Callable
MaskFn = tp.Any


def and_masks(*mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) & result
        return result

    return mask


def or_masks(*mask_fns):
    def mask(b, h, q, kv):
        result = jnp.zeros((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) | result
        return result

    return mask


def vmap_bhqkv(mask_fn, with_head=False):
    fn = jax.vmap(mask_fn, in_axes=(None, None, None, 0))  # kv arange
    fn = jax.vmap(fn, in_axes=(None, None, 0, None))  # q arange
    if with_head:
        fn = jax.vmap(fn, in_axes=(None, 0, None, None))  # h arange
    fn = jax.vmap(fn, in_axes=(0, None, None, None))

    return fn


def causal_mask_function(b, h, q, kv):
    return q >= kv


def sliding_window_mask_overlay(window_size: int):
    def mask(b, h, q, kv):
        return jnp.where(q - kv >= 0, q - kv <= window_size, kv - q <= window_size)

    return mask


def sliding_window_causal_overlay(window_size: int):
    """Overlay depicting a sliding-window pattern for causal attention.

    Mirrors HF's semantics: kv_idx > q_idx - window_size (exclusive bound).
    """

    def mask(b, h, q, kv):
        return kv > q - window_size

    return mask


def document_mask_overlay(segment_ids: Int[Array, "..."]):
    def mask(b, h, q, kv):
        return segment_ids[b, q] == segment_ids[b, kv]

    return mask


def ignore_padding_overlay(segment_ids: Int[Array, "..."], pad_id: int = 0):
    def mask(b, h, q, kv):
        return segment_ids[b, kv] != pad_id

    return mask


def dummy_mask_function(b, h, q, kv):
    return True


def make_bool_interface(
    q_length: int,
    kv_length: int,
    mask_function: MaskFn,
    padding_mask: Bool[Array, "B T"] | None = None,
    batch_size: int | None = None,
    nheads: int | None = None,
) -> Bool[Array, "B T S"] | Bool[Array, "B N T S"] | Bool[Array, "T S"]:
    batch_arange = jnp.arange(batch_size, dtype=jnp.int32)
    q_arange = jnp.arange(q_length, dtype=jnp.int32)
    kv_arange = jnp.arange(kv_length, dtype=jnp.int32)
    head_arange = jnp.arange(nheads, dtype=jnp.int32) if nheads else None

    mask_output = vmap_bhqkv(
        mask_function,
        with_head=head_arange is not None,
    )(batch_arange, head_arange, q_arange, kv_arange)

    mask_ndim = mask_output.ndim
    if padding_mask is not None:
        if mask_ndim == 3:  # (B, T, S), (B, T) -> (B, 1, T, S)
            return (mask_output & padding_mask[:, :, None]).astype(jnp.bool)[
                :, None, :, :
            ]
        elif mask_ndim == 4:  # (B, N, T, S), (B, T) -> (B, N, T, S)
            return (mask_output & padding_mask[:, None, :, None]).astype(jnp.bool)
    else:
        return mask_output.astype(jnp.bool)[:, None, :, :]


class AttentionMaskInterface(GeneralInterface[str, MaskImpl]):
    _global_mapping = {
        "eager": make_bool_interface,
        "sdpa": make_bool_interface,
        "xla_chunked": make_bool_interface,
        "chunked_manual": make_bool_interface,
    }


ATTENTION_MASK_INTERFACE = AttentionMaskInterface()


def make_causal_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """
    Generates a mask for causal attention.
    Right now, only support for training.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B #N T T"] or _BlockMask
        The computed causal attention mask,
    """

    # Need to think more about maskign for inference but whatever
    B, T, H = input_embeds.shape

    mask_interface = ATTENTION_MASK_INTERFACE[mask_impl]
    mask_factory_function = causal_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    causal_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return causal_mask


def make_bidirectional_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T D"],
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
    **kwargs,
) -> Bool[Array, "B T S"] | BlockMask | Bool[Array, "B T N S"]:
    """
    Generates a mask for bidirectional attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T T"] or _BlockMask
        The computed full attention mask,
    """

    B, T, H = input_embeds.shape

    mask_interface = ATTENTION_MASK_INTERFACE[mask_impl]
    mask_factory_function = dummy_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            dummy_mask_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    full_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return full_mask


def slliding_window_full_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    window_size: int,
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """
    Generates a mask for sliding window attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    window_size : int
        Size of the sliding window.
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T T"] or _BlockMask
        The computed sliding window attention mask,
    """
    B, T, H = input_embeds.shape

    mask_interface = ATTENTION_MASK_INTERFACE[mask_impl]
    mask_factory_function = and_masks(
        sliding_window_mask_overlay(window_size), dummy_mask_function
    )

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    sliding_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return sliding_mask


def make_sliding_window_causal_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    window_size: int,
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """Generates a sliding-window causal mask.

    Equivalent to `make_causal_mask` with an additional sliding-window overlay.
    """

    B, T, _H = input_embeds.shape
    mask_interface = ATTENTION_MASK_INTERFACE[mask_impl]
    mask_factory_function = and_masks(
        sliding_window_causal_overlay(window_size),
        causal_mask_function,
    )

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    sliding_causal_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return sliding_causal_mask
