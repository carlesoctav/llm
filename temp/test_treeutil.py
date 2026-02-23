from jaxformers.print_utils import tree_pprint
import jax.numpy as jnp
import jax
import equinox as eqx


tree = {"a": jnp.ones((10, 10)), "b": 10}



left, right = eqx.partition(tree)
tree_pprint(left)
tree_pprint(right)
jax.tree.flatten
