"""JAX Guided RTC for the pinned pi0.5 sampler (noise time 1 -> 0).

Reference: PI real-time-chunking-kinetix/src/model.py, realtime_action.
The correction differentiates the denoised estimate with respect to x_t,
not with respect to model weights. Only six physical output dimensions
are constrained. No training-time prefix conditioning is assumed.
"""
import jax
import jax.numpy as jnp


def prefix_weights(delay, valid, horizon=50, soft_steps=None):
    end = valid if soft_steps is None else jnp.minimum(valid, delay + soft_steps)
    start = jnp.minimum(delay, end)
    positions = jnp.arange(horizon)
    linear = jnp.clip(1 + (start - 1 - positions) / (end - start + 1), 0, 1)
    tapered = linear * jnp.expm1(linear) / jnp.expm1(1.)
    tapered = jnp.where(positions < start, 1., jnp.clip(tapered,0.,1.))
    return jnp.where(positions < end, tapered, 0.)


def corrected_velocity(velocity_fn, x, time, previous, weights, beta):
    def estimate(z):
        velocity = velocity_fn(z)
        return z - time * velocity, velocity
    clean, pullback, velocity = jax.vjp(estimate, x, has_aux=True)
    residual = (previous - clean) * weights
    correction = pullback(residual)[0]
    # Convert the paper's data-time tau to OpenPI's noise-time t=1-tau.
    tau = 1. - time
    gain = jnp.minimum((time * time + tau * tau) /
                       jnp.maximum(time * tau, 1e-8), beta)
    return velocity - gain * correction


def sample_guided(model, rng, observation, previous, weights, *, num_steps=10, beta=1.0):
    """Match OpenPI's KV-cached sampler, adding VJP guidance in the Euler loop."""
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    _, cache = model.PaliGemma.llm([prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1)

    def velocity(x, t):
        suffix, suffix_mask, suffix_ar, conditioning = model.embed_suffix(
            observation, x, jnp.broadcast_to(t, (batch,)))
        prefix_attention = jnp.broadcast_to(prefix_mask[:, None, :],
            (batch, suffix.shape[1], prefix.shape[1]))
        attention = jnp.concatenate([prefix_attention, make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, output), _ = model.PaliGemma.llm([None, suffix], mask=attention,
            positions=positions, kv_cache=cache, adarms_cond=[None, conditioning])
        # Checkpoint projections may emit bf16; the Euler latent and VJP
        # correction are fp32. Both cond branches must share the latent dtype.
        return model.action_out_proj(output[:, -model.action_horizon:]).astype(x.dtype)

    noise = jax.random.normal(rng, (batch, model.action_horizon, model.action_dim))
    dt = -1. / num_steps

    def step(x, i):
        time = 1. + i * dt
        v = jax.lax.cond(jnp.any(weights > 0),
            lambda z: corrected_velocity(lambda y: velocity(y, time), z, time, previous, weights, beta),
            lambda z: velocity(z, time), x)
        return x + dt * v, None

    actions, _ = jax.lax.scan(step, noise, jnp.arange(num_steps))
    return actions
