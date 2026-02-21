import jax
import equinox as eqx
eqx.partition
import jax.tree_util as jtu
jtu.DictKey



key = jax.random.key(10)
weights = {}
for i in "abcdef":
    inside_key = jax.random.fold_in(key, i)
    weights[i] = jax.random(key, (100, 100))


def make_mask(weights):
    def f(path, leaf):
        if jtu.keystr(path, simple = True)  in "ab":
            return True
        else:
            return False
    return jtu.tree_map_with_path(f, weights)


train_mask = make_mask(weights)
