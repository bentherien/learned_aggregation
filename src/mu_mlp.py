import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base

from collections.abc import Iterable
from typing import Any, Mapping, Tuple, Callable, Optional
from learned_optimization.tasks import base
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask, find_smallest_divisor
from haiku._src.typing import Initializer

from helpers import MupVarianceScaling
import globals
from mu_task_base import MuTask

State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray
Batch = Any

class MuMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      input_mult=1.0,
      output_mult=1.0,
      hidden_lr_mult=1.0,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs an MLP with MuP following table 8 of tensor programs V.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    self.get_adam_mup_lr_mul = {}
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      if index ==0:
        #input layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w':1.0,'b':1.0}
        
      elif index == len(output_sizes) - 1:
        #output layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=jnp.zeros,# RandomNormal(stddev=1., mean=0.),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w':1.0,'b':1.0}
      else:
        #hidden layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w': hidden_lr_mult / output_sizes[index-1] ,'b':1.0}
        
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None
    
    assert len(output_sizes) >= 2, "need more than one layer for MuMLP"
    

    self.input_mult = input_mult
    self.hidden_mult = 1.0
    self.output_mul =  output_mult * 1 / output_sizes[-2]
    hk.set_state("mup_lrs", self.get_adam_mup_lr_mul)

  @property
  def mup_lrs(self):
    return hk.get_state("mup_lrs")
  
  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i == 0:
        out = out * self.input_mult
      elif i < (num_layers - 1):
        out = out * self.hidden_mult

      # hk.set_state("layer_%d_act_l1" % i, jnp.mean(jnp.abs(out)))
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out * self.output_mul
        

  def reverse(
      self,
      activate_final: Optional[bool] = None,
      name: Optional[str] = None,
  ) -> "MLP":
    """Returns a new MLP which is the layer-wise reverse of this MLP.

    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once.

    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.

    >>> mlp = hk.nets.MLP([1, 2, 3])
    >>> mlp_in = jnp.ones([1, 2])
    >>> y = mlp(mlp_in)
    >>> rev = mlp.reverse()
    >>> rev_mlp_out = rev(y)
    >>> mlp_in.shape == rev_mlp_out.shape
    True

    Args:
      activate_final: Whether the final layer of the MLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.

    Returns:
      An MLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self.activate_final
    if name is None:
      name = self.name + "_reversed"

    output_sizes = tuple(
        layer.input_size
        for layer in reversed(self.layers)
        if layer.input_size is not None
    )
    if len(output_sizes) != len(self.layers):
      raise ValueError("You cannot reverse an MLP until it has been called.")
    return MuMLP(
        output_sizes=output_sizes,
        w_init=self.w_init,
        b_init=self.b_init,
        with_bias=self.with_bias,
        activation=self.activation,
        activate_final=activate_final,
        name=name,
    )


class _MuMLPImageTask(_MLPImageTask, MuTask):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0,
               mup_multipliers=dict(input_mult=1.0,
                                    output_mult=1.0,
                                    hidden_lr_mult=1.0),):
    super().__init__(
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0)
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets
    self.mup_state = None

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return MuMLP( #hk.nets.MLP(
          sizes, activation=act_fn,
              **mup_multipliers)(
              inp, dropout_rate=dropout_rate, 
              rng=hk.next_rng_key())

    self._mod = hk.transform_with_state(_forward)
    
    self.mup_state = None
    self.init_mup_state()

  def loss_with_state(self, params, state, key, data):
    num_classes = self.datasets.extra_info["num_classes"]
    logits, state = self._mod.apply(params, state, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss), self.get_mup_state(state)

  def init_with_state(self, key: PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._mod.init(key, batch["image"])
    return params, self.get_mup_state(state)

  def loss_and_accuracy_with_state(self, params: Params, state: State, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_classes = self.datasets.extra_info["num_classes"]

    threshold = 10000
    if data["image"].shape[0] > threshold:
      # If the batch is too large, we split it into smaller chunks.
      # This is to avoid running out of memory.
      # This is not necessary for the task to work, but it is useful for
      # large batch sizes.
      data["image"] = jnp.array_split(data["image"], 
                                      find_smallest_divisor(data["image"].shape[0],threshold), 
                                      axis=0)
      # print(jax.tree_map(lambda x: x.shape, data["image"]))
      # exit(0)
      to_cat = [self._mod.apply(params, state, key, chunk)[0] for chunk in data["image"]]

      logits = jnp.concatenate(to_cat)
    else:
      logits = self._mod.apply(params, state, key, data["image"])[0]
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy
  


  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    # if state is not None:
      # raise ValueError("Define a custom loss_with_state_and_aux when using a"
      #                  " state!")
    aux = {}
    loss, state = self.loss_with_state(params, state, key, data)
    return loss, state, aux
  

