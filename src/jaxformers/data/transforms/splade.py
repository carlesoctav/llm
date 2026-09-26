import dataclasses as dc
import typing as tp

import numpy as np
from grain import transforms as grain_transforms
from transformers import PreTrainedTokenizerBase


@dc.dataclass
class TokenizeSplade(grain_transforms.Map):
    """Tokenize query/positive/negatives with SPLADE instruction prefixes."""

    tokenizer: PreTrainedTokenizerBase
    query_column: str
    positive_column: str
    negatives_column: str | None
    num_negatives: int
    query_prefix: str
    document_prefix: str
    query_max_length: int
    doc_max_length: int

    def _encode(self, text: str, prefix: str, max_length: int) -> dict[str, np.ndarray]:
        encoded = self.tokenizer(
            prefix + text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        pooling_mask = attention_mask
        if prefix:
            offsets = np.asarray(encoded["offset_mapping"]).squeeze(0)
            specials = (
                np.asarray(encoded["special_tokens_mask"]).squeeze(0).astype(bool)
            )
            in_prefix = (offsets[:, 0] < len(prefix)) & ~specials
            pooling_mask = np.where(in_prefix, 0, pooling_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pooling_mask": pooling_mask,
        }

    def map(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.query_column not in features:
            raise KeyError(f"Column {self.query_column!r} not found in element")
        if self.positive_column not in features:
            raise KeyError(f"Column {self.positive_column!r} not found in element")
        negatives = []
        if self.num_negatives > 0:
            if self.negatives_column not in features:
                raise KeyError(f"Column {self.negatives_column!r} not found in element")
            negatives = features[self.negatives_column]
            if isinstance(negatives, str):
                negatives = [negatives]
            negatives = list(negatives)
            if len(negatives) < self.num_negatives:
                raise ValueError(
                    f"Column {self.negatives_column!r} has {len(negatives)} negatives "
                    f"but num_negatives={self.num_negatives}"
                )
            negatives = negatives[: self.num_negatives]
        query = self._encode(
            features[self.query_column], self.query_prefix, self.query_max_length
        )
        docs = [
            self._encode(text, self.document_prefix, self.doc_max_length)
            for text in [features[self.positive_column], *negatives]
        ]
        return {
            "query": query,
            "docs": {key: np.stack([doc[key] for doc in docs]) for key in query},
        }


def make(
    tokenizer: PreTrainedTokenizerBase | None = None,
    query_column: str = "query",
    positive_column: str = "positive",
    negatives_column: str | None = "negatives",
    num_negatives: int = 1,
    query_prefix: str = "[Q] ",
    document_prefix: str = "[D] ",
    query_max_length: int = 128,
    doc_max_length: int = 512,
) -> list[grain_transforms.Map]:
    """Build the list of transforms required for SPLADE contrastive training."""

    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if num_negatives > 0 and negatives_column is None:
        raise ValueError("negatives_column is required when num_negatives > 0")

    return [
        TokenizeSplade(
            tokenizer=tokenizer,
            query_column=query_column,
            positive_column=positive_column,
            negatives_column=negatives_column,
            num_negatives=num_negatives,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            query_max_length=query_max_length,
            doc_max_length=doc_max_length,
        )
    ]
