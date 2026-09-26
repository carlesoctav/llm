import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import ModernBertConfig

from jaxformers.models.huggingface.modernbert import ModernBertForMaskedLM
from jaxformers.module_utils import AdditionalConfig, ForwardImpl


class SpladeModel(ModernBertForMaskedLM):
    vocab_fold_index: Array

    def __init__(
        self,
        config: ModernBertConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
        store_config: bool = True,
    ):
        super().__init__(
            config,
            additional_config,
            rngs=rngs,
            param_dtype=param_dtype,
            store_config=store_config,
        )
        self.vocab_fold_index = jnp.arange(config.vocab_size, dtype=jnp.int32)

    def _fold(self, sparse: Float[Array, "B V"]) -> Float[Array, "B V"]:
        return jax.vmap(
            lambda row: jnp.zeros_like(row).at[self.vocab_fold_index].max(row)
        )(sparse)

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        dtype: jnp.dtype = jnp.float32,
        *,
        attention_mask: Bool[Array, "B T"] | None = None,
        pooling_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        logits = super().__call__(
            input_ids,
            dtype,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            rngs=rngs,
            forward_impl=forward_impl,
            **inputs,
        )
        weights = jnp.log1p(jax.nn.relu(logits - self.config.logit_shift))
        position_top_k = self.config.position_top_k
        if position_top_k is not None and position_top_k < self.config.vocab_size:
            cutoff = jax.lax.top_k(weights, position_top_k)[0][..., -1:]
            weights = weights * (weights >= cutoff)
        if pooling_mask is None:
            pooling_mask = attention_mask
        if pooling_mask is not None:
            weights = weights * pooling_mask[..., None].astype(weights.dtype)
        sparse = weights.max(axis=1)
        if self.config.vocab_fold is not None:
            sparse = self._fold(sparse)
        return sparse

    @staticmethod
    def score(
        queries: Float[Array, "Nq V"], documents: Float[Array, "Nd V"]
    ) -> Float[Array, "Nq Nd"]:
        return queries @ documents.T
