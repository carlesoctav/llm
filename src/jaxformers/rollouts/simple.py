from jaxformers.distributed.parallel import mutate_sharding_rule_parallel_dims
import typing as tp
from functools import partial

import jax
import jax.tree_util as jtu
import dataclasses
from jaxtyping import PRNGKeyArray




# @partial(
#     jtu.register_dataclass,
#     data_fields=[],
#     meta_fields=[],
# )
@jtu.register_dataclass
@dataclasses.dataclass
class SamplingState:
    key: jax.Array
    step: int | jax.Array  # [B]
    tokens: jax.Array  # [B, T + decode_chunk_length * loop]
    token_logprobs: jax.Array  # [batch, decode_state_length+1], [:, 0] is dummy
    # token_scores: jax.Array  # [batch, decode_state_length+1], [:, 0] is dummy
    kv: jax.Array  # [B, T + decode_chunk_length * loop, ...]
    done: jax.Array  # [B]
    max_decode_step: jax.Array  # [B]
    eos_ids: jax.Array  # [n_eos]

@dc.dataclass(frozen=True)
class RolloutParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_decode_steps: int = 256
    decode_chunk_size: int | None = None
    max_seq_len: int = 1025
    max_input_len: int | None = None
    num_samples_per_example: int = 1
    min_prefill_size: int = 256
    prefill_size: int | None = None
    stop_token_ids: tuple[int, ...] | list[int] = ()

    def get_decoding_schedule(
        self,
        *,
        min_input_length: int,
        max_input_length: int,
    ) -> DecodingSchedule:
        prefill_size = self.prefill_size
        if prefill_size is None:
            prefill_size = max(
                int(np.exp2(np.ceil(np.log2(max(min_input_length, 1))))),
                self.min_prefill_size,
            )

        begin_position = min(prefill_size, max(min_input_length - 1, 0))
        end_position = min(
            self.max_seq_len - 1,
            int(max_input_length) + self.max_decode_steps - 1,
        )
        chunk_size = self.decode_chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = self.max_decode_steps

        return DecodingSchedule(
            prefill_size=prefill_size,
            begin_position=begin_position,
            end_position=end_position,
            chunk_size=chunk_size,
        )

class SimpleRollout:
    def __init__(
        self,
        model: Model
        parallel_dims: ParallelDims,
        devices: list | None = None,
    ):
        self.tokenizer = model.tokenizer
        new_config = model.config
        mutate_sharding_rule_parallel_dims()
        self.forward = partial(model.forward)
        self.mesh = jax.make_mesh()

    def generate(
        input_text: list[dict[str, tp.Any]],
        weights,
        *,
        prefill_size: int = -1,
        sampling_params: SamplingParams | None = None,
        scoring_params: ScoringParams | None = None,
        include_eos_in_output_text: bool = False,
        scoring_inputs: bool = True,
        rngs: PNRGKeyArray = None
    )
