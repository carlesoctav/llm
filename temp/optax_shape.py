from jaxformers.optimizers import adam
import jax
from jaxformers.callback.log_learning_rate import log_learning_rate

w = jax.random.normal(jax.random.key(10), (12, 1000))
tx = adam.make(1e-5, 1, 1.0)
opt_state = tx.init(w)

fn = log_learning_rate()
callback_state = fn.init(w, opt_state)
print("DEBUGPRINT {callback_state}:", callback_state)
