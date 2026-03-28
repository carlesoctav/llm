from typing import Callable, NamedTuple


class Callback(NamedTuple):
    init: Callable  # (model)
    update: Callable  # (model, callback_state, grad, updates, aux) -> (model, callback_state)
    process: Callable  # (output, model, callback_state, aux) -> (output, model, callback_state)


def callback_chain(*args: Callback) -> Callback:
    if not args:
        raise ValueError("callback_chain requires at least one callback")

    init_fns, update_fns, process_fns = zip(*args)

    def init_fn(model):
        return tuple(fn(model) for fn in init_fns)

    def update_fn(model, callback_state, grad, updates, aux):
        new_state = []
        for s, fn in zip(callback_state, update_fns):
            model, new_s = fn(model, s, grad, updates, aux)
            new_state.append(new_s)

        return model, tuple(new_state)

    def process_fn(output, model, callback_state, aux):
        new_state = []
        for s, fn in zip(callback_state, process_fns):
            output, model, new_s = fn(output, model, s, aux)
            new_state.append(new_s)
        return output, model, tuple(new_state)

    return Callback(init_fn, update_fn, process_fn)
