from jaxformers.print_utils import tree_pprint
import functools

import jax

from jaxformers import tree_util


def make_scan_fwd(fwd, num_hidden_layers):
    def scan_fwd(x, w, input_kwargs):
        @functools.wraps(fwd)
        def fn(carry, x):
            inputs = carry
            scan_weights, scan_input_kwargs = x
            input_kwargs = tree_util.combine(scan_input_kwargs, nonscan_input_kwargs)
            inputs = fwd(inputs, scan_weights, **input_kwargs)
            return inputs, None

        scan_weights, _ = split_scan_items(w, num_hidden_layers)
        scan_input_kwargs, nonscan_input_kwargs = split_scan_items(input_kwargs, num_hidden_layers)
        print("DEBUGPRINT {nonscan_input_kwargs}:", nonscan_input_kwargs)
        print("DEBUGPRINT {scan_input_kwargs}:", scan_input_kwargs)
        carry, _ = jax.lax.scan(
            fn,
            init = x,
            xs = (scan_weights, scan_input_kwargs)
        )
        return carry
    return scan_fwd

def split_scan_items(weights, size, index=0):
    def filter(leaf):
        if isinstance(leaf, jax.Array):
            if leaf.shape == ():
                return False
            return leaf.shape[index] == size
        return False
    scan_items, other_items = tree_util.partition(weights, filter)
    return scan_items, other_items
