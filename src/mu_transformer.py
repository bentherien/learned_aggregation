# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Haiku modules for transformers.

This is a fork of the transformer from the haiku examples. This transformer
implementation is very minimal and thus will not achive peak performance.
It is, however, relatively simple and can be implemented in <200 LOC.
"""

from typing import Any, Mapping, Tuple, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any, Mapping

import chex
from learned_optimization.tasks import base
from mu_task_base import MuTask

State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray
Batch = Any

class MuCausalSelfAttention(hk.Module):
  """Multi-headed attention mechanism.

  As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
      hidden_lr_mult: float = 1.0,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads

    if not w_init:
      w_init = hk.initializers.VarianceScaling(1.)

    self.w_init = w_init

    self._imput_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._hidden_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._output_w_init = jnp.zeros

    # the bias is an input weight whose input dimension is always 1
    self._b_init = hk.initializers.RandomNormal(stddev=1., mean=0.)
    adam_lr_mul = {'w': hidden_lr_mult / (self.key_size * self.num_heads), 'b': 1}
    hk.set_state("mup_lrs",{'linear': adam_lr_mul,
                            'query': adam_lr_mul,
                            'key':adam_lr_mul,
                            'value': adam_lr_mul,})

  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Compute (optionally masked) MHA with queries, keys & values."""

    key = key if key is not None else query
    value = value if value is not None else query

    if query.ndim != 3:
      raise ValueError("Expect queries of shape [B, T, D].")

    seq_len = query.shape[1]
    causal_mask = onp.tril(onp.ones((seq_len, seq_len)))
    mask = mask * causal_mask if mask is not None else causal_mask

    query_heads = self._mu_linear_projection(query, self.key_size, w_init=self._hidden_w_init,
                                             b_init=self._b_init, name="query")
    
    key_heads = self._mu_linear_projection(key, self.key_size, w_init=self._hidden_w_init, 
                                           b_init=self._b_init, name="key")
    
    value_heads = self._mu_linear_projection(value, self.value_size, w_init=self._hidden_w_init,
                                             b_init=self._b_init, name="value")

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    sqrt_key_size = self.key_size # onp.sqrt(self.key_size).astype(key.dtype)
    attn_logits = attn_logits / sqrt_key_size

    if mask is not None:
      if mask.ndim != attn_logits.ndim:
        raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                         f"{attn_logits.ndim}.")
      attn_logits = jnp.where(mask, attn_logits, -1e30)

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    # Concatenate attention matrix of all heads into a single vector.
    attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

    return hk.Linear(self.model_size, 
                     w_init=self._hidden_w_init,
                     b_init=self._b_init)(attn_vec)

  @hk.transparent
  def _mu_linear_projection(self,
                            x: jnp.ndarray,
                            head_size: int,
                            w_init: Optional[hk.initializers.Initializer] = None,
                            b_init: Optional[hk.initializers.Initializer] = None,
                            name: Optional[str] = None) -> jnp.ndarray:
    # hidden layer type
    y = hk.Linear(self.num_heads * head_size,
                  w_init=w_init,
                  b_init=b_init,
                  name=name)(x)
    
    return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class MuDenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               d_model,
               widening_factor: int = 4,
               w_init: Optional[hk.initializers.Initializer] = None,
               b_init: Optional[hk.initializers.Initializer] = None,
               name: Optional[str] = None,
               hidden_lr_mult: float = 1.0,):
    super().__init__(name=name)

    #hidden layer init
    self._w_init = w_init
    self._b_init = b_init
    self._widening_factor = widening_factor

    hk.set_state("mup_lrs",{'linear':  {'w': hidden_lr_mult / d_model, 'b': 1},
                            'linear_1':  {'w': hidden_lr_mult / (4 * d_model), 'b': 1},})

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    x = hk.Linear(self._widening_factor * hiddens,
                  w_init=self._w_init,
                  b_init=self._b_init,
                  name='linear')(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens,
                      w_init=self._w_init,
                      b_init=self._b_init,
                      name='linear_1')(x)


class MuTransformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               d_model: int,
               vocab_size: int,
               dropout_rate: float,
               name: Optional[str] = None,
               input_mult: float = 1.0,
               output_mult: float = 1.0,
               hidden_lr_mult: float = 1.0,):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._d_model = d_model
    self._vocab_size = vocab_size
    self._output_mult = output_mult / d_model
    self._input_mult = input_mult
    self._hidden_lr_mult = hidden_lr_mult

    self._imput_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._hidden_w_init = hk.initializers.VarianceScaling(1.0, "fan_in",  "normal")
    self._output_w_init = jnp.zeros

    # the bias is an input weight whose input dimension is always 1
    self._b_init = hk.initializers.RandomNormal(stddev=1., mean=0.)


    adam_lr_mul_in = {'embeddings': 1}
    adam_lr_mul_out = {'w': 1, 'b': 1}
    adam_lr_mul_ln = {'scale': 1, 'offset': 1}
    mup_lrs = {'embed_in': adam_lr_mul_in,
               'linear_out': adam_lr_mul_out,
               'h_f': adam_lr_mul_ln,}
    
    for i in range(self._num_layers):
      mup_lrs[f"h{i}_ln_1"] = adam_lr_mul_ln
      mup_lrs[f"h{i}_ln_2"] = adam_lr_mul_ln


    hk.set_state("mup_lrs",mup_lrs)

    assert d_model % num_heads == 0, "Number of heads must divide model size."


  # def get_mup_lrs(self,params):
  #   jax.tree_map(lambda x: x, params)

  def __call__(self, h: jnp.ndarray, mask: Optional[jnp.ndarray],
               is_training: bool) -> jnp.ndarray:
    """Connects the transformer.

    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.

    Returns:
      Array of shape [B, T, D].
    """
    h = hk.Embed(vocab_size=self._vocab_size, 
                 embed_dim=self._d_model, 
                 w_init=self._imput_w_init,
                 name='embed_in'
                 )(h)
    
    #apply MuP input Multiplier
    if self._input_mult != 1.0:
      h = h * self._input_mult

    init_scale = 2. / self._num_layers

    dropout_rate = self._dropout_rate if is_training else 0.

    if mask is not None:
      mask = mask[:, None, None, :]

    for i in range(self._num_layers):

      h_norm = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True, name=f"h{i}_ln_1",
          scale_init=jnp.ones,offset_init=jnp.zeros)(h)
      
      h_attn = MuCausalSelfAttention(
          num_heads=self._num_heads,
          key_size=self._d_model // self._num_heads,
          model_size=h.shape[-1],
          name=f"h{i}_attn",
          hidden_lr_mult=self._hidden_lr_mult)(
            h_norm, mask=mask)
      
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)

      h = h + h_attn

      h_norm = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True, name=f"h{i}_ln_2",
          scale_init=jnp.ones,offset_init=jnp.zeros)(h)
      
      h_dense = MuDenseBlock(d_model=self._d_model,
                             name=f"h{i}_mlp",
                             w_init=self._hidden_w_init,
                             b_init=self._b_init,
                             hidden_lr_mult=self._hidden_lr_mult)(h_norm)
      
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)

      h = h + h_dense

    h = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name="h_f",
        scale_init=jnp.ones,offset_init=jnp.zeros)(h)

    return hk.Linear(self._vocab_size,
                     w_init=self._output_w_init,
                     b_init=self._b_init,
                     name='linear_out')(h) * self._output_mult



class _MuTransformerTask(base.Task, MuTask):
  """Tranformer from a dictionary configuration."""

  def __init__(self, datasets, cfg: Mapping[str, Any], name: str = '__TransformerTask',
               mup_multipliers=dict(input_mult=1.0,
                                    output_mult=1.0,
                                    hidden_mult=1.0)):
    self.mup_multipliers = mup_multipliers
    self.datasets = datasets
    self._cfg = cfg
    self._net = hk.transform_with_state(self._hk_forward)
    self._name = name

    self.mup_state = None
    self.init_mup_state()

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    vocab_size = self.datasets.extra_info['vocab_size']
    mod = MuTransformer(
        num_heads=self._cfg['num_heads'],
        num_layers=self._cfg['num_layers'],
        d_model=self._cfg['d_model'],
        dropout_rate=self._cfg['dropout_rate'],
        vocab_size=vocab_size,
        **self.mup_multipliers)
    mask = (batch['image'] != 0)
    logits = mod(batch['image'], mask=mask, is_training=True)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], vocab_size))
    return jnp.sum(loss * mask) / jnp.sum(mask)

  def init(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def init_with_state(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._net.init(key, batch)
    return params, self.get_mup_state(state)

  def loss(self, params, key, data):
    return self._net.apply(params, key, data)

  def loss_with_state(self, params, state, key, data):
    params, state = self._net.apply(params, state, key, data)
    return params, self.get_mup_state(state)
  


  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    aux = {}
    loss, state = self.loss_with_state(params, state, key, data)
    return loss, state, aux