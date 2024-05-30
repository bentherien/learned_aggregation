
import pickle
import os.path as osp
from learned_optimization import checkpoints

from glob import glob
import os
import re
import csv
import numpy as np
from functools import reduce
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# CHECKPOINT SAVING HELPERS



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def delete_old_checkpoints(save_dir, n_to_keep):
    ckpt_dir_regex = r"global_step[\d]*"
    if save_dir.endswith("/"):
        save_dir = save_dir.strip("/")
    all_ckpts = natural_sort(
        [
            i
            for i in glob(f"{save_dir}/*")
            if i.endswith(".ckpt") and re.search(ckpt_dir_regex, i)
        ]
    )
    all_pkl = natural_sort(
        [
            i
            for i in glob(f"{save_dir}/*")
            if i.endswith(".pickle") and re.search(ckpt_dir_regex, i)
        ]
    )

    n_to_delete = len(all_ckpts) - n_to_keep
    if n_to_delete > 0:
        to_delete_ckpt = all_ckpts[:n_to_delete]
        to_delete_pkl = all_pkl[:n_to_delete]
        print(
            f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete_ckpt + to_delete_pkl)}"
        )
        for ckpt in to_delete_ckpt + to_delete_pkl:
            try:
                os.remove(ckpt)
            except FileNotFoundError:
                pass


def save_checkpoint(
    prefix, i, args, outer_trainer_state
):  # Checkpoint every 1000th iteration
    save_dir = osp.join("checkpoints", prefix + args.meta_train_name)
    checkpoints.save_state(
        osp.join(
            save_dir,
            "global_step{}.ckpt".format(i + 1),
        ),
        outer_trainer_state,
    )
    pickle_filename = osp.join(
        save_dir,
        "global_step{}.pickle".format(i + 1),
    )
    with open(
        pickle_filename,
        "wb",
    ) as f:
        pickle.dump(
            outer_trainer_state.gradient_learner_state.theta_opt_state.params, f
        )

    with open(osp.join(save_dir, "latest"), "w") as f:
        f.write("global_step{}".format(i + 1))

    delete_old_checkpoints(
        save_dir=save_dir,
        n_to_keep=args.checkpoints_to_keep,
    )

    return pickle_filename


def get_ckpt_dirs(ckpt_dir, meta_train_name):
    a = os.listdir(ckpt_dir)
    keep = []
    for x in a:
        if osp.isdir(osp.join(ckpt_dir, x)) and x[8:] == meta_train_name:
            keep.append(x)
    return keep


def get_ckpt_to_load(ckpt_dir, dirs):
    def nat_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key[1])]
        return sorted(l, key=alphanum_key)

    sortable = []
    for x in dirs:
        if osp.isfile(osp.join(ckpt_dir, x, "latest")):
            ckpt = open(osp.join(ckpt_dir, x, "latest"), "r").readline().strip()
            sortable.append(
                (
                    osp.join(ckpt_dir, x, ckpt),
                    ckpt,
                )
            )
    sortable = nat_sort(sortable)

    keep = []
    for x in sortable:
        if x[1] == sortable[-1][1]:
            keep.append(x)
    if len(keep) > 1:
        print(
            "[Warning] multiple directories contain a checkpoint at the same latest iteration. Selecting one arbitrarily."
        )

    return keep[0]


def get_resume_ckpt(ckpt_dir, meta_train_name):
    dirs = get_ckpt_dirs(ckpt_dir, meta_train_name)
    if len(dirs) == 0:
        print("[Info] No existing checkpoint found. Starting from scratch.")
        return None
    ckpt_path, suffix = get_ckpt_to_load(ckpt_dir, dirs)
    print("[Info] Loading checkpoint from {}".format(ckpt_path))
    return ckpt_path




def save_timings_to_csv(timings, filename, column_name):
    """
    Saves the timings to a CSV file.

    :param timings: List of execution times.
    :param filename: Name of the file to save the data.
    :param column_name: Name of the column under which timings are saved.
    """
    # Calculate and print the average timing
    average_timing = sum(timings) / len(timings)
    print(f"Average timing: {average_timing} seconds")

    # Save the timings to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([column_name])  # Write the header
        for timing in timings:
            writer.writerow([timing])




is_leaf = lambda x : reduce(np.logical_and, [type(x1) != dict for x1 in x.values()])

def add_prefix(prefix,s):
    if prefix != '':
        prefix = prefix + '/'
    return prefix + s

def get_mup_lrs_hk(state,prefix):
    d = {}
    for k,v in state.items():
        if is_leaf(v):
            d[add_prefix(prefix,k)] = v
        else:
            for kk,vv in get_mup_lrs_hk(v,k).items():
                d[add_prefix(prefix,kk)] = vv
    
    d = {k.replace('/mup_lrs',''):v for k,v in d.items()}
    return d

def get_mup_lrs_from_state(state):
    if 'flax_mup_lrs' in state:
        lrs = state['flax_mup_lrs']
    else:
        lrs = get_mup_lrs_hk({k:{'mup_lrs':v['mup_lrs']} for k,v in state.items() if 'mup_lrs'in v.keys()}, 
                             prefix='')
    

    return lrs






from typing import Any


import haiku as hk
import jax
import jax.numpy as jnp

State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray

from collections.abc import Sequence

from haiku._src.typing import Initializer
import numpy as np
from haiku.initializers import * 
  



def _compute_fans(shape, fan_in_axes=None):
  """Computes the number of input and output units for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    if fan_in_axes is not None:
      # Compute fan-in using user-specified fan-in axes.
      fan_in = np.prod([shape[i] for i in fan_in_axes])
      fan_out = np.prod([s for i, s in enumerate(shape)
                         if i not in fan_in_axes])
    else:
      # If no axes specified, assume convolution kernels (2D, 3D, or more.)
      # kernel_shape: (..., input_depth, depth)
      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

class MupVarianceScaling(hk.initializers.Initializer):
  """Initializer which adapts its scale to the shape of the initialized array.

  The initializer first computes the scaling factor ``s = scale / n``, where n
  is:

    - Number of input units in the weight tensor, if ``mode = fan_in``.
    - Number of output units, if ``mode = fan_out``.
    - Average of the numbers of input and output units, if ``mode = fan_avg``.

  Then, with ``distribution="truncated_normal"`` or ``"normal"``,
  samples are drawn from a distribution with a mean of zero and a standard
  deviation (after truncation, if used) ``stddev = sqrt(s)``.

  With ``distribution=uniform``, samples are drawn from a uniform distribution
  within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.

  The variance scaling initializer can be configured to generate other standard
  initializers using the scale, mode and distribution arguments. Here are some
  example configurations:

  ==============  ==============================================================
  Name            Parameters
  ==============  ==============================================================
  glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
  glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
  lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
  lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
  he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
  he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
  ==============  ==============================================================
  """

  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal',
               fan_in_axes=None):
    """Constructs the :class:`VarianceScaling` initializer.

    Args:
      scale: Scale to multiply the variance by.
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``normal`` or ``uniform``.
      fan_in_axes: Optional sequence of int specifying which axes of the shape
        are part of the fan-in. If none provided, then the weight is assumed
        to be like a convolution kernel, where all leading dimensions are part
        of the fan-in, and only the trailing dimension is part of the fan-out.
        Useful if instantiating multi-headed attention weights.
    """
    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.fan_in_axes = fan_in_axes

  def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape, self.fan_in_axes)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      # distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      # stddev = stddev / distribution_stddev
      return TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      return RandomUniform(minval=-limit, maxval=limit)(shape, dtype)





def set_non_hashable_args(args):
    if args.run_type in ["benchmark", "sweep"]:
        if args.optimizer in ['small_fc_mlp', 'mup_small_fc_mlp', 'adamw', 'velo', 'muadam']:
            args.meta_testing_batch_size = args.local_batch_size
            args.batch_shape = (args.local_batch_size,)
            args.label_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices)))
            args.image_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,1,1,1)))
        else:
            args.batch_shape = (args.num_grads * args.num_local_steps * args.local_batch_size,)
            args.label_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices)))
            args.image_sharding = PositionalSharding(mesh_utils.create_device_mesh((args.num_devices,1,1,1))) 

            args.meta_testing_batch_size = args.num_grads \
                                            * args.num_local_steps \
                                            * args.local_batch_size
    else:
        
        if args.optimizer == 'small_fc_mlp' or args.optimizer == 'mup_small_fc_mlp':
            args.batch_shape = (args.steps_per_jit, args.num_tasks, args.local_batch_size)
            args.label_sharding = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices)))
            args.image_sharding = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices,1,1,1)))
            args.meta_training_batch_size = args.local_batch_size \
                                            * args.num_tasks \
                                            * args.steps_per_jit
        else:
            args.batch_shape = (args.steps_per_jit, args.num_tasks, args.num_grads * args.num_local_steps * args.local_batch_size)
            args.label_sharding = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices)))
            args.image_sharding = PositionalSharding(mesh_utils.create_device_mesh((1,1,args.num_devices,1,1,1)))

            args.meta_training_batch_size = args.num_grads \
                                            * args.num_local_steps \
                                            * args.local_batch_size \
                                            * args.num_tasks \
                                            * args.steps_per_jit
    return args