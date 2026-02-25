import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


def logsumexp_pshard(lse_local, axis):
    m = lax.pmax(lse_local, axis)
    return m + jnp.log(lax.psum(jnp.exp(lse_local - m), axis))


def ce_vocab_sharded_blocked(
    x, labels, w, *, b_block, v_block, h_block, dtype=jnp.float32, precision=None
):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("tp",))

    V = w.shape[1]
    tp = mesh.shape["tp"]
    Vpad = (-V) % (tp * v_block)
    w = jnp.pad(
        w, ((0, 0), (0, Vpad))
    )  # so each shard gets equal V_shard divisible by v_block

    @shard_map(
        mesh=mesh,
        in_specs=(
            P(None, None),
            P(None),
            P(None, "tp"),
        ),  # x/labels replicated, w sharded on vocab
        out_specs=(P(None), P(None)),
        check_rep=False,
    )
    def per_shard(x, labels, w_shard):
        B, H = x.shape
        V_shard = w_shard.shape[1]
        shard_start = lax.axis_index("tp") * V_shard

        num_b = B // b_block
        num_h = H // h_block
        num_v_local = V_shard // v_block

        loss = jnp.zeros((B,), dtype)
        lse = jnp.zeros((B,), dtype)

        # ---- B LOOP ----
        def b_body(bi, carry):
            loss, lse = carry
            b0 = bi * b_block
            xb = lax.dynamic_slice(x, (b0, 0), (b_block, H))
            yb = lax.dynamic_slice(labels, (b0,), (b_block,)).astype(jnp.int32)

            lse_local = jnp.full(
                (b_block,), -jnp.inf, dtype
            )  # logsumexp over THIS shard's vocab slice
            ll_local = jnp.full(
                (b_block,), -jnp.inf, dtype
            )  # label logit if label is in THIS shard

            # ---- V LOOP (LOCAL TO SHARD) ----
            @jax.checkpoint  # (your remat goes here)
            def v_body(vi, state):
                lse_local, ll_local = state
                v0 = vi * v_block

                # ---- H LOOP (OPTIONAL) ----
                def h_body(hi, acc):
                    h0 = hi * h_block
                    x_tile = lax.dynamic_slice(xb, (0, h0), (b_block, h_block))
                    w_tile = lax.dynamic_slice(w_shard, (h0, v0), (h_block, v_block))
                    return acc + lax.dot_general(
                        x_tile,
                        w_tile,
                        (((1,), (0,)), ((), ())),
                        precision=precision,
                        preferred_element_type=jnp.float32,
                    )

                logits = lax.fori_loop(
                    0, num_h, h_body, jnp.zeros((b_block, v_block), jnp.float32)
                ).astype(dtype)

                v0_global = shard_start + v0
                valid = (v0_global + jnp.arange(v_block)) < V
                logits = jnp.where(valid, logits, -jnp.inf)

                lse_local = jnp.logaddexp(lse_local, jax.nn.logsumexp(logits, axis=-1))

                in_block = (yb >= v0_global) & (yb < v0_global + v_block)
                idx = jnp.where(in_block, yb - v0_global, 0)
                gathered = logits[jnp.arange(b_block), idx]
                ll_local = jnp.where(in_block, gathered, ll_local)
                return lse_local, ll_local

            lse_local, ll_local = lax.fori_loop(
                0, num_v_local, v_body, (lse_local, ll_local)
            )

            lse_b = logsumexp_pshard(lse_local, "tp")
            in_shard = (yb >= shard_start) & (yb < shard_start + V_shard)
            ll_b = lax.psum(jnp.where(in_shard, ll_local, 0.0), "tp")

            loss_b = lse_b - ll_b
            loss = lax.dynamic_update_slice(loss, loss_b, (b0,))
            lse = lax.dynamic_update_slice(lse, lse_b, (b0,))
            return loss, lse

        return lax.fori_loop(0, num_b, b_body, (loss, lse))

    return per_shard(x, labels, w)
