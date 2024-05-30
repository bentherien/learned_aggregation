
"""
The following file defines tasks.

Tasks follow the following naming conventions:
MODEL_DATASET

for language modelling:
    DATASET = {dataset name}-s{sequence length}-v{vocab size}

for image datasets:
    DATASET = {dataset name}-{HxWxC}

for MLPs:
    MODEL = {model name}-w{width}-d{depth}

"""

import gin
import jax
import functools
import ml_collections
from typing import Tuple
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple
from learned_optimization import profile

from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.datasets.base import Datasets, ThreadSafeIterator, LazyIterator, _image_map_fn
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask
from learned_optimization.tasks.fixed.transformer_lm import _TransformerTask
from learned_optimization.tasks.fixed.vit import (VisionTransformerTask, deit_tiny_config, deit_small_config)
from learned_optimization.tasks.resnet import ResNet
from learned_optimization.tasks.fixed.resnet import _ResnetTaskDataset

from mu_vit import MuVisionTransformerTask

# from fast_imagenet import fast_imagenet_datasets
from mu_mlp import _MuMLPImageTask
from mu_transformer import _MuTransformerTask
from learned_optimization.tasks.datasets.language import _make_datasets, get_32k_sentence_piece_vocab
import functools

from flax.training import prefetch_iterator
import jax
import jax.numpy as jnp
from learned_optimization import profile
import numpy as onp
import tensorflow_datasets as tfds
import warnings

Batch = Any

import os
import time
import h5py
import io
from PIL import Image
import multiprocessing


class Timer:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Executing {self.func.__name__} took {end_time - start_time:.4f} seconds.")
        return result


def process_batch(encoded_images):
    """Process a batch of encoded images into numpy arrays."""
    return [onp.array(Image.open(io.BytesIO(img_data)).convert('RGB')) for img_data in encoded_images]

class H5Data:
    _instance = None

    @Timer
    def __new__(cls, h5_path, num_workers=24):
        if cls._instance is None:
            print("Creating the dataset instance")
            cls._instance = super(H5Data, cls).__new__(cls)
            
            # Read the encoded images and labels from the H5 file
            with h5py.File(h5_path, 'r') as file:
                encoded_images = file['encoded_images'][:]
                targets = file['targets'][:]
            
            # Determine the number of workers
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            
            # Create batches of encoded images
            batch_size = len(encoded_images) // num_workers
            image_batches = [encoded_images[i:i + batch_size] for i in range(0, len(encoded_images), batch_size)]

            # Use multiprocessing to process the batches
            with multiprocessing.Pool(num_workers) as pool:
                image_arrays = pool.map(process_batch, image_batches)
            
            # Flatten the list of lists to a single list
            cls._instance.data = onp.array([img for sublist in image_arrays for img in sublist])
            cls._instance.labels = onp.squeeze(targets)

        return cls._instance
    
    
def parse_split(split_string, data_array, index_array):
    # Extract the range from the string after removing 'train[' and ']'
    range_part = split_string[len('train['):-1]

    # Split the range on ':'
    parts = range_part.split(':')
    num_samples = len(data_array)
    
    # Determine start index
    start = parts[0].strip()
    if start.endswith('%'):
        start_index = int(float(start.rstrip('%')) / 100 * num_samples)
    else:
        start_index = int(start) if start else 0

    # Determine end index
    end = parts[1].strip() if len(parts) > 1 else ''
    if end.endswith('%'):
        end_index = int(float(end.rstrip('%')) / 100 * num_samples)
    else:
        end_index = int(end) if end else num_samples

    # Return the appropriate slice of the data array
    return data_array[start_index:end_index], index_array[start_index:end_index]

    
class PreloadImageNetDatasetH5():
    
    def __init__(self, split, h5_path, num_workers):
        self.split = split
        self.h5_path = h5_path
        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = 100000
        self.data = H5Data(h5_path=h5_path, num_workers=num_workers)
        
    def preload(self):
        im, lab = self.preload_helper()
        return {'image':im, 'label':lab}
    
    def preload_helper(self):
        if self.split.startswith('train'):
            s = self.split.split('train')[-1]
            if s == '':
                return H5Data._instance.data[:self.n_train], H5Data._instance.labels[:self.n_train]
            else:
                return parse_split(split_string=self.split, 
                                   data_array=H5Data._instance.data[:self.n_train], 
                                   index_array=H5Data._instance.labels[:self.n_train])
        elif self.split.lower() == 'validation':
            return H5Data._instance.data[self.n_train:self.n_train + self.n_val], H5Data._instance.labels[self.n_train:self.n_train + self.n_val]
        elif self.split.lower() == 'test':
            return H5Data._instance.data[self.n_train + self.n_val:], H5Data._instance.labels[self.n_train + self.n_val:]
        else:
            raise NotImplemented('not implemented for split'+str(self.split))
        


def custom_preload_tfds_image_classification_datasets(
    datasetname: str,
    h5_path: str,
    splits: Tuple[str, str, str, str],
    batch_size: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: Tuple[int, int, int, int] = (20,1,1,1),
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None,
    convert_to_black_and_white: Optional[bool] = False,
    batch_shape=None,
    label_sharding=None,
    image_sharding=None,
) -> Datasets:
  """Load an image dataset with tfds by first loading into host ram.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs
    convert_to_black_and_white: conver a color image to black and white.

  Returns:
    A Datasets object containing data iterators.
  """
  assert len(splits) == len(prefetch_batches), 'number of splits and prefetch_batches should be the same'
  assert len(splits) == len(batch_size), 'number of splits and batch_size should be the same'
  prefetch_batches = {splits[i]:prefetch_batches[i] for i in range(len(splits))}
  batch_size = {splits[i]:batch_size[i] for i in range(len(splits))}

  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": False,
      "aug_flip_up_down": False,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std,
      "convert_to_black_and_white": convert_to_black_and_white,
  }

  def make_python_iter(split: str) -> Iterator[Batch]:
    # load the entire dataset into memory
    with profile.Profile(f"tfds.load({datasetname})"):
        #   dataset = _cached_tfds_load(datasetname, split=split, batch_size=-1)
      dataset = PreloadImageNetDatasetH5(split, h5_path=h5_path, num_workers=24)
      dataset = dataset.preload()
    data = tfds.as_numpy(_image_map_fn(cfg, dataset))

    # import pdb; pdb.set_trace()
    print(jax.tree_map(lambda x:x.shape if type(x) != int else x, data))

    # use a python iterator as this is faster than TFDS.
    def generator_fn():

      def iter_fn():
        
        if batch_size[split] > data["image"].shape[0]:
          warnings.warn('For {} split {}, batch size ({}) is larger than dataset size ({}). Possible'
                ' duplicate samples in batch/'.format(
                    datasetname,split,batch_size[split],data["image"].shape[0]), Warning)
          batches = 1
          idx = onp.arange(batch_size[split]) % data["image"].shape[0]
        else:
          batches = data["image"].shape[0] // batch_size[split]
          idx = onp.arange(data["image"].shape[0])

        if 'train' in split:
            print('using infinite iterator for training')
            #infinite iterator
            while True:
                idxs = onp.random.permutation(idx)[:batch_size[split]]
                # onp.random.choice(range(0, data["image"].shape[0]), size=batch_size[split], replace=False)
                
                def index_into(idxs, x):
                  #TODO fix hacky label check
                  sharding = image_sharding if len(x.shape) > 1 else label_sharding
                  temp_batch_shape = batch_shape + x.shape[1:] if len(batch_shape) > 1 \
                                           else (batch_size[split],) + x.shape[1:]

                  return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)



                yield jax.tree_util.tree_map(functools.partial(index_into, idxs), data)
        else:
            print('using epoch based iterator for testing')
            while True:
              # every epoch shuffle indicies
              onp.random.shuffle(idx)

              for bi in range(0, batches):
                idxs = idx[bi * batch_size[split]:(bi + 1) * batch_size[split]]

                def index_into(idxs, x):
                  #TODO fix hacky label check
                  sharding = image_sharding if len(x.shape) > 1 else label_sharding
                  temp_batch_shape = batch_shape + x.shape[1:] if len(batch_shape) > 1 \
                                           else (batch_size[split],) + x.shape[1:]

                  return jnp.reshape(jax.device_put(x[idxs], device=sharding), temp_batch_shape)



                yield jax.tree_util.tree_map(functools.partial(index_into, idxs), data)
      
      return prefetch_iterator.PrefetchIterator(iter_fn(), prefetch_batches[split])
      
    return ThreadSafeIterator(LazyIterator(generator_fn))

  builder = tfds.builder(datasetname)
  num_classes = builder.info.features["label"].num_classes

  if stack_channels == 1:
    output_channel = builder.info.features["image"].shape[-1:]
  else:
    output_channel = (stack_channels,)

  if convert_to_black_and_white:
    output_channel = (1,)

  abstract_batch = {
      "image":
          jax.core.ShapedArray(
              (batch_size[splits[0]],) + image_size + output_channel, dtype=jnp.float32),
      "label":
          jax.core.ShapedArray((batch_size[splits[0]],), dtype=jnp.int32)
  }
  return Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes, 'name':datasetname},
      abstract_batch=abstract_batch)


@base.dataset_lru_cache
@gin.configurable
def imagenet_64_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    # prefetch_batches=[20,1,1,1],
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:

    assert image_size in [(32,32),(64,64),(128,128),(225,225)]
    h5_path = os.path.join(os.environ["TFDS_DATA_DIR"],'imagenet_{}x{}x3_JPEG.h5'.format(image_size[0],image_size[1]))
    perc = max(1, int(80 * data_fraction))
    splits = (f"train[0:{perc}%]", "train[97%:98%]", "train[98:99%]", "validation")
    # splits = (f"train[0:50%]", "train[50%:99%]", "train[50%:99%]", "validation")
    return custom_preload_tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        h5_path=h5_path,
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        # prefetch_batches=prefetch_batches,
        # shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )


# @gin.configurable
# def mlp128x128_fastinet_32(batch_size):
#     """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
#     h5_path = "/mnt/raid0/imagenet_hdf5/ilsvrc2012.hdf5"
#     datasets = fast_imagenet_datasets(h5_path, 
#         batch_size, 
#         workers=48, 
#         distributed=False,
#         image_size=(32,32,),
#         output_channel=(3,)
#     )
#     return _MLPImageTask(datasets, [128, 128])



# @base.dataset_lru_cache
# @gin.configurable
# def imagenet_64_datasets(
#     batch_size: int,
#     image_size: Tuple[int, int] = (64, 64),
#     # prefetch_batches=[20,1,1,1],
#     data_fraction=1.0,
#     **kwargs,
# ) -> base.Datasets:
#     perc = max(1, int(80 * data_fraction))
#     splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
#     return base.preload_tfds_image_classification_datasets(
#         datasetname="imagenet_resized",
#         splits=splits,
#         batch_size=batch_size,
#         image_size=image_size,
#         stack_channels=1,
#         # prefetch_batches=prefetch_batches,
#         # shuffle_buffer_size=10000,
#         normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
#         normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
#         convert_to_black_and_white=False,
#         # cache=True,
#         **kwargs,
#     )

def func_create_func(task_fun, ds_args, model_args):
    ds_fun = ds_args['fun']
    # model_fun = model_args['fun']
    print('[func_create_func]',model_args)
    print('[func_create_func]',ds_args)
    return task_fun(ds_fun(*ds_args['args'],**ds_args['kwargs']),**model_args)


def add_MLP_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for mlp_width in widths:
            for mlp_depth in depths:
                tasks['mlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, _MLPImageTask, ds,
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth))
                
                tasks['mumlp-w{}-d{}_{}'.format(mlp_width,mlp_depth,k)] = functools.partial(func_create_func, 
                                                                                            _MuMLPImageTask, 
                                                                                            ds,
                                                                                            dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                 mup_multipliers=mup_muls))

def add_sweepable_MLP_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        for mlp_width in widths:
            for mlp_depth in depths:
                for iname,input_mult in [('2**2',2**2)]:
                    for oname,output_mult in [('2**5',2**5)]:
                        for hname,hidden_mult in [('2**1',2**1),('2**-2',2**-2),('2**-4',2**-4),('2**-3',2**-3),('2**5',2**-5),('2**6',2**-6),('2**8',2**-8)]:
                        
                            tasks['mumlp-w{}-d{}-i{}-o{}-h{}_{}'.format(mlp_width,mlp_depth,iname,oname,hname,k)] = functools.partial(func_create_func, 
                                                                                                                                        _MuMLPImageTask, 
                                                                                                                                        ds,
                                                                                                                                        dict(hidden_sizes=[mlp_width] * mlp_depth,
                                                                                                                                        output_mult=output_mult,
                                                                                                                                        input_mult=input_mult,
                                                                                                                                        hidden_mult=hidden_mult))
                
def add_transformer_lm_tasks(tasks, lm_datasets, widths, depths, mup_muls):
    for k,ds in lm_datasets.items():
        for w,heads in widths:
            for d in depths:
                cfg = {
                    "num_heads": heads,
                    "d_model": w,
                    "num_layers": d,
                    "dropout_rate": 0.1,
                }
                name = 'transformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _TransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name))
                
                name = 'mutransformer-w{}-d{}_{}'.format(w,d,k)
                tasks[name] = functools.partial(func_create_func, 
                                                _MuTransformerTask, 
                                                ds,
                                                dict(cfg=cfg,name=name,
                                                     mup_multipliers=mup_muls))


def vit_w64_d3():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 64
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 64 * 4
  config.transformer.num_heads = 4
  config.transformer.num_layers = 3
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


def deit_tiny_config():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w192_d3():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 3
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w192_d8():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 8
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w192_d16():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 16
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w192_d32():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 32
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w192_d64():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 64
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def deit_small_config():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 384 * 4
  config.transformer.num_heads = 6
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def vit_w1024_d3():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 1024
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1024 * 4
  config.transformer.num_heads = 8
  config.transformer.num_layers = 3
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


def vit_w2048_d3():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 2048
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 2048 * 4
  config.transformer.num_heads = 16
  config.transformer.num_layers = 3
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config

def get_h14_config():
  """Returns the ViT-H/14 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-H_14'
  config.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.hidden_size = 1280
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 5120
  config.transformer.num_heads = 16
  config.transformer.num_layers = 32
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config

def get_h14_config_small():
  """Returns the ViT-H/14 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-H_14'
  config.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.hidden_size = 1280
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 5120
  config.transformer.num_heads = 16
  config.transformer.num_layers = 8
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config

def add_vision_transformer_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        # for w,heads in widths:
        #     for d in depths:
        w=384
        d=12
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=deit_small_config()))
        

        w=192
        d=12
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=deit_tiny_config()))
        

        w=1024
        d=3
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w1024_d3()))
        

        w=2048
        d=3
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w2048_d3()))
        

        w=192
        d=3
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d3(),
                                             mup_multipliers=mup_muls))
        
        w=192
        d=3
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d3()))
        

        w=192
        d=8
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d8(),
                                             mup_multipliers=mup_muls))
        
        w=192
        d=8
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d8()))
        
        w=192
        d=16
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d16(),
                                             mup_multipliers=mup_muls))
        
        w=192
        d=16
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d16()))
        
        w=192
        d=32
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d32(),
                                             mup_multipliers=mup_muls))
        
        w=192
        d=32
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w192_d32()))
        
        w=192
        d=64
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask,
                                        ds,
                                        dict(cfg=vit_w192_d64(),
                                             mup_multipliers=mup_muls))
        
        w=192
        d=64
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask,
                                        ds,
                                        dict(cfg=vit_w192_d64()))
        

        w=2048
        d=3
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w2048_d3(),
                                             mup_multipliers=mup_muls))
        

        w=1024
        d=3
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w1024_d3(),
                                             mup_multipliers=mup_muls))
        
        w=1280
        d=32
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=get_h14_config(),
                                             mup_multipliers=mup_muls))
        
        w=1280
        d=32
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=get_h14_config()))
        

        w=1280
        d=8
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=get_h14_config_small()))
        
        w=64
        d=3
        name = 'vit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        VisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w64_d3()))
        
        w=64
        d=3
        name = 'muvit-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        MuVisionTransformerTask, 
                                        ds,
                                        dict(cfg=vit_w64_d3(),
                                             mup_multipliers=mup_muls))
        
        
        


def add_resnet_tasks(tasks, image_datasets, widths, depths, mup_muls):
    for k,ds in image_datasets.items():
        # for w,heads in widths:
        #     for d in depths:
        cfg = dict(initial_conv_kernel_size=7,
                initial_conv_stride=2,
                resnet_v2=False, 
                max_pool=True,
                **ResNet.CONFIGS[50])
        w=64
        d=50
        name = 'resnet50-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ResnetTaskDataset, 
                                        ds,
                                        dict(cfg=cfg))
        

        cfg = dict(initial_conv_kernel_size=7,
                initial_conv_stride=2,
                resnet_v2=False, 
                max_pool=True,
                **ResNet.CONFIGS[18])
        w=256
        d=18
        name = 'resnet50-w{}-d{}_{}'.format(w,d,k)
        tasks[name] = functools.partial(func_create_func, 
                                        _ResnetTaskDataset, 
                                        ds,
                                        dict(cfg=cfg))


def get_test_batch_size(task):
    if 'cifar' in task:
        return 10000
    elif 'food101' in task:
        return 10000
    elif 'fashionmnist' in task:
        return 10000
    elif 'imagenet' in task:
        return 50000
    elif 'lm1b' in task:
        return 128
    else:
        raise ValueError(f"Unknown task: {task}")
                
def get_task(args, is_test=False):

    mup_multipliers = dict(
        input_mult=args.mup_input_mult,
        output_mult=args.mup_output_mult,
        hidden_lr_mult=args.mup_hidden_lr_mult
    )

    created_tasks = []
    for chosen_task in args.task:

        if args.run_type in ['benchmark','sweep']:
            batch_size = (args.meta_testing_batch_size,1,1,get_test_batch_size(chosen_task),)
            prefetch_batches = (2,1,1,2)
        else:
            if args.meta_loss_split is not None:
                batch_size = (args.meta_training_batch_size,1,args.meta_training_batch_size,1,)
                prefetch_batches = (args.prefetch_batches,1,args.prefetch_batches,1)
            else:
                batch_size = (args.meta_training_batch_size,1,1,1,)
                prefetch_batches = (args.prefetch_batches,1,1,1)


        ds_kwargs = dict(prefetch_batches=prefetch_batches,
                            batch_shape=args.batch_shape,
                            label_sharding=args.label_sharding,
                            image_sharding=args.image_sharding,)


        IMAGE_DATASET_REGISTY = {
            'imagenet-32x32x3': dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(32, 32), **ds_kwargs)),
            'imagenet-64x64x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(64, 64), **ds_kwargs)),
            'imagenet-128x128x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(128, 128), **ds_kwargs)),
            'imagenet-224x224x3':  dict(fun=imagenet_64_datasets,args=[],kwargs=dict(batch_size=batch_size,image_size=(224, 224), **ds_kwargs)),

            'cifar10-32x32x3': dict(fun=image.cifar10_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'food101-32x32x3': dict(fun=image.food101_datasets,args=[],kwargs=dict(batch_size=batch_size, image_size=(32, 32), **ds_kwargs)),
            'fashionmnist-28x28x1': dict(fun=image.fashion_mnist_datasets,args=[],kwargs=dict(batch_size=batch_size, **ds_kwargs)),
        }

        LANGUAGE_DATASET_REGISTY = {
            'lm1b-s2048-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=2048, **ds_kwargs)),
            'lm1b-s1024-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=1024, **ds_kwargs)),
            'lm1b-s512-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=512, **ds_kwargs)),
            'lm1b-s256-v32k': dict(fun=_make_datasets,args=['lm1b',],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=256, **ds_kwargs)),
            'lm1b-s128-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=128, **ds_kwargs)),
            'lm1b-s64-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=64, **ds_kwargs)),
            'lm1b-s32-v32k': dict(fun=_make_datasets,args=['lm1b', ],kwargs=dict(vocab='sentencepiece', batch_size=batch_size, sequence_length=32, **ds_kwargs)),
        }
        
        tasks = {}

        add_MLP_tasks(tasks, 
                      image_datasets=IMAGE_DATASET_REGISTY, 
                      widths=[2**i for i in range(16)], 
                      depths=[3,6,8,12,16,24,32,64],
                      mup_muls=mup_multipliers)
        
        add_sweepable_MLP_tasks(tasks, 
                                image_datasets=IMAGE_DATASET_REGISTY, 
                                widths=[128,512], 
                                depths=[3],
                                 mup_muls=mup_multipliers)

        add_transformer_lm_tasks(tasks, 
                                 lm_datasets=LANGUAGE_DATASET_REGISTY,
                                 widths=[(64,2),(128,2),(192,3),(384,6),(768,12),(1024,8),(2048,16),(4096,32)], 
                                 depths=[3,6,8,12,16,24,32,64],
                                 mup_muls=mup_multipliers)
        
        add_vision_transformer_tasks(tasks,
                                     image_datasets=IMAGE_DATASET_REGISTY, 
                                     widths=[], 
                                     depths=[],
                                     mup_muls=mup_multipliers)


        add_resnet_tasks(tasks, 
                         image_datasets=IMAGE_DATASET_REGISTY, 
                         widths=[], 
                         depths=[],
                         mup_muls=mup_multipliers)
        
        created_tasks.append(tasks[chosen_task]())
    
    if len(created_tasks) == 1:
        return created_tasks[0]
    else:
        return created_tasks