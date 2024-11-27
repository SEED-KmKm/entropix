from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from entropix.dslider import DSState, adaptive_dirichlet_step, initialize_state
from entropix.dslider_config import DSConfig, DEFAULT_DS_CONFIG


MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def debug_print_e(fmt: str, *args, **kwargs):
  jax.debug.callback(
      lambda *args, **kwargs: print(fmt.format(*args, **kwargs), end=""),
      *args, **kwargs)

@dataclass
class SamplerConfig:
  # Naked (logits) entropy thresholds
  low_naked_entropy_threshold = 0.1  # Captures most observed LELV cases
  medium_naked_entropy_threshold = 1.2  # Separates medium from high entropy cases
  high_naked_entropy_threshold = 2.5  # Above this we see clear high entropy cases

  # Naked (logits) varentropy thresholds
  low_naked_varentropy_threshold = 1.2  # Most LELV cases are below this
  high_naked_varentropy_threshold = 2.5  # Clear separation for high variance cases

  # Scaffold (attention) metrics thresholds
  # These don't appear in logs, keeping unchanged
  low_scaffold_entropy_threshold = 1.0
  high_scaffold_entropy_threshold = 2.0
  low_scaffold_varentropy_threshold = 0.3
  high_scaffold_varentropy_threshold = 0.8

@partial(jax.jit, static_argnames=("cfg",))
def sample(
  state: DSState,
  logits: jnp.ndarray,
  cfg: DSConfig,
  clarifying_question_token: int = 2564,
  key=jax.random.PRNGKey(1337),
) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array]]:
  sample_cfg = SamplerConfig()
  bsz = logits.shape[0]
  # breakpoint()
  (
    new_state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    naked_token_logprob,
    scaffold_token_logprob,
  ) = adaptive_dirichlet_step(key, state, logits, cfg)
  new_token = new_token.reshape((bsz, 1))

  def _and(*args):
    res = True
    for a in args:
      res = jax.lax.bitwise_and(res, a)
    return res

  def sample_one(
    idx,
    logit,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    loops=0,
  ):
    LELV = _and(
      naked_ent < sample_cfg.low_naked_entropy_threshold,
      naked_varent < sample_cfg.low_naked_varentropy_threshold,
      # scaffold_ent < sample_cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < sample_cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    HELV = _and(
      naked_ent > sample_cfg.high_naked_entropy_threshold,
      naked_varent < sample_cfg.low_naked_varentropy_threshold,
      # scaffold_ent < sample_cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < sample_cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    LEHV = _and(
      naked_ent < sample_cfg.high_naked_entropy_threshold,
      naked_varent > sample_cfg.high_naked_varentropy_threshold,
      # scaffold_ent < sample_cfg.low_scaffold_entropy_threshold,
      # scaffold_varent > sample_cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    HEHV = _and(
      naked_ent > sample_cfg.medium_naked_entropy_threshold,
      naked_varent > sample_cfg.high_naked_varentropy_threshold,
      # scaffold_ent > sample_cfg.high_scaffold_entropy_threshold,
      # scaffold_varent > sample_cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    DEFO=1-(LELV+HELV+LEHV+HEHV)

    case = jnp.argmax(jnp.hstack([LELV, HELV, LEHV, HEHV, DEFO]))

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    def lelv():
      # jax.debug.print("LELV Naked Ent: {}", naked_ent)
      # jax.debug.print("LELV Naked Varent: {}", naked_varent)
      # jax.debug.print("LELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LELV Scaffold Varent: {}\n", scaffold_varent)
      # jax.debug.print("[lelv]")
      # jax.debug.breakpoint()
      return new_token, state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    def helv():
      # jax.debug.print("HELV Naked Ent: {}", naked_ent)
      # jax.debug.print("HELV Naked Varent: {}", naked_varent)
      # jax.debug.print("HELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HELV Scaffold Varent: {}\n", scaffold_varent)
      # jax.debug.print("[helv]")
      # jax.debug.breakpoint()
      return jnp.array([2564]), state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    def lehv():
      # jax.debug.print("LEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("LEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("LEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LEHV Scaffold Varent: {}\n", scaffold_varent)
      # TODO(xjdr): We need to do a differnt version of tree search here with constant return dimensions
      # jax.debug.print("[lehv]")
      # jax.debug.breakpoint()
      return new_token, state

    # High Entropy, High Varentropy: "resampling in the mist"
    def hehv():
      # jax.debug.print("HEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("HEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("HEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HEHV Scaffold Varent: {}\n", scaffold_varent)
      # jax.debug.print("[hehv]")
      # jax.debug.breakpoint()
      plogit = logit.at[new_token].set(float("-inf"))

      # Run ADS with single batch
      (
        new_state,
        resampled_token,
        *_,  # Other metrics
      ) = adaptive_dirichlet_step(
        key,
        jax.tree_map(lambda x: x[None, ...], state),
        plogit[None, ...],  # Shape (1, vocab)
        DEFAULT_DS_CONFIG,
      )
      return resampled_token, jax.tree_map(lambda x: jnp.bfloat16(x[-1]), new_state)

    def default():
      # jax.debug.print("Default Naked Ent: {}", naked_ent)
      # jax.debug.print("Default Naked Varent: {}", naked_varent)
      # jax.debug.print("[dflt]")
      # jax.debug.breakpoint()
      return new_token, state
    
    debug_print_e("[c{}]", case)
    # jax.debug.breakpoint()
    return jax.lax.switch(case, (lelv, helv, lehv, hehv, default))

  result, new_state = jax.vmap(sample_one)(
    jnp.arange(bsz),
    logits,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
  )
  samplestats={'ent':naked_ent, 'varent':naked_varent}
  return result.reshape((bsz, 1)), new_state, samplestats
