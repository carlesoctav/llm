import jax

@jax.jit
def f(w, x):
    return w @ x


def main():
    key = jax.random.key(10)
    key1, key2 = jax.random.split(key)
    w = jax.random.normal(key1, (1, 10))
    x = jax.random.normal(key2, (10, 1))
    trace = f.trace(w, x).lower().compile().input_formats
    print("DEBUGPRINT {trace}:", trace)


main()
