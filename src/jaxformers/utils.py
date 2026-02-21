import typing as tp
from typing import Any, Callable

import wadler_lindig as wl
from attr import dataclass


K = tp.TypeVar("K")
V = tp.TypeVar("V")


class GeneralInterface(tp.MutableMapping[K, V], tp.Generic[K, V]):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key: K) -> V:
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: tp.Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())
