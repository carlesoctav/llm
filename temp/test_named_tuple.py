from typing import NamedTuple

import jax.tree_util as jtu


class Test(NamedTuple):
    learning_rate: float


mapping = {"model_1": [10, Test(learning_rate=10)], "model_2": Test(learning_rate=20)}


def f(path, leaf):
    print(path)


is_tuple = lambda x: isinstance(x, Test)
jtu.tree_map_with_path(f, mapping, is_leaf=is_tuple)
jtu.tree_map_with_path(f, mapping, is_leaf=is_tuple)
