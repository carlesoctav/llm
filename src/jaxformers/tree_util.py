from jax.tree_util import KeyPath, SequenceKey, DictKey, GetAttrKey, FlattenedIndexKey, KeyEntry

def optimizerstr(keys: KeyPath, separator: str = '/') -> str:
  str_fn = _optimizer_entrystr
  return separator.join(map(str_fn, keys))

def _optimizer_entrystr(key: KeyEntry) -> str:
    match key:
        case (
            DictKey(key=key)
            | GetAttrKey(name=key)
            | FlattenedIndexKey(key=key)
        ):
            return str(key)
        case SequenceKey(idx=key):
            return ""
        case _:
            return str(key)
