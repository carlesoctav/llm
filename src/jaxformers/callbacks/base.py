from typing import Callable, NamedTuple


class Callback(NamedTuple):
    init: Callable  # (weights, opt_state)
    update: Callable  # (grad, updates, opt_state, weights, aux) -> new callbackState #inside jit
    process: Callable  # (logger)


def callback_chain(*args: Callback) -> Callback:
    init_fns, update_fns, process_fns = zip(*args)

    init_fns, update_fns, process_fns = zip(*args)
    def init_fn(weights, opt_state):
        return tuple(fn(weights, opt_state) for fn in init_fns)

    def update_fn(callback_state, grad, updates, opt_state, weights, aux):
        new_state = []
        for s, fn in zip(callback_state, update_fns):
            new_s = fn(s, grad, updates, opt_state, weights, aux)
            new_state.append(new_s)

        return tuple(new_state)

    def process_fn(output, callback_state, aux):
        new_state = []
        for s, fn in zip(callback_state, process_fns):
            output, new_s = fn(output, callback_state, aux)
        new_state.append(new_s)
        return output, tuple(new_state)

    return Callback(init_fn, update_fn, process_fn)
