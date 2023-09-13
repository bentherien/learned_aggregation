import gin
import jax
from typing import Tuple
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask


@base.dataset_lru_cache
@gin.configurable
def imagenet_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> base.Datasets:
    splits = ("train", "validation", "validation", "test")
    return base.tfds_image_classification_datasets(
        datasetname="imagenet2012",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=50,
        shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        **kwargs,
    )


@base.dataset_lru_cache
@gin.configurable
def imagenet_64_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    **kwargs,
) -> base.Datasets:
    splits = ("train", "validation", "validation", "validation")
    return base.tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=50,
        shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        **kwargs,
    )


@gin.configurable
def My_Conv_Food101_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=101)
    datasets = image.food101_datasets(batch_size=batch_size)
    return _ConvTask(base_model_fn, datasets)

@gin.configurable
def My_Conv_Imagenet_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss(
        [32, 64, 64], jax.nn.relu, num_classes=1000
    )
    datasets = imagenet_datasets(batch_size=batch_size)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_Conv_Imagenet64_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss(
        [32, 64, 64], jax.nn.relu, num_classes=1000
    )
    datasets = imagenet_64_datasets(batch_size=batch_size)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_Conv_Cifar10_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=batch_size, prefetch_batches=5)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_Conv_Cifar10_8_16x32(batch_size):
    """A 2 hidden layer convnet designed for 8x8 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _ConvTask(base_model_fn, datasets)

@gin.configurable
def My_ImageMLP_Imagenet_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = imagenet_64_datasets(batch_size=batch_size,
                                    image_size=(32, 32),)
    return _MLPImageTask(datasets, [128, 128])

@gin.configurable
def My_ImageMLP_Cifar10_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.cifar10_datasets(batch_size=batch_size,
                                      prefetch_batches=50,)
    return _MLPImageTask(datasets, [128, 128])

@gin.configurable
def My_ImageMLP_FashionMnist_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size, prefetch_batches=5)
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def My_ImageMLP_FashionMnist_Relu64x64(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [64, 64])


@gin.configurable
def My_ImageMLP_FashionMnist_Relu32x32(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [32, 32])


@gin.configurable
def My_Conv_FashionMnist_28_16x32(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_ImageMLP_FashionMnist8_Relu32(batch_size):
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32])


def get_task(args, is_test=False):
    tasks = {
        "image-mlp-fmst": My_ImageMLP_FashionMnist_Relu128x128,
        "image-mlp-fmst64x64": My_ImageMLP_FashionMnist_Relu64x64,
        "image-mlp-fmst32x32": My_ImageMLP_FashionMnist_Relu32x32,
        "small-image-mlp-fmst": My_ImageMLP_FashionMnist8_Relu32,
        "image-mlp-c10-128x128": My_ImageMLP_Cifar10_Relu128x128,
        "conv-c10": My_Conv_Cifar10_32x64x64,
        "small-conv-c10": My_Conv_Cifar10_8_16x32,
        "conv-imagenet64": My_Conv_Imagenet64_32x64x64,
        "conv-imagenet": My_Conv_Imagenet_32x64x64,
        "image-mlp-imagenet32-128x128": My_ImageMLP_Imagenet_Relu128x128,
        "fmnist-conv-mlp-mix": [My_Conv_FashionMnist_28_16x32,
                       My_ImageMLP_FashionMnist_Relu64x64,
                       My_ImageMLP_FashionMnist_Relu128x128],
        "fmnist-mlp-mix": [My_ImageMLP_FashionMnist_Relu32x32,
                       My_ImageMLP_FashionMnist_Relu64x64,
                       My_ImageMLP_FashionMnist_Relu128x128],
        "dataset-mlp-mix": [My_ImageMLP_Imagenet_Relu128x128,
                            My_ImageMLP_Cifar10_Relu128x128,
                            My_ImageMLP_FashionMnist_Relu128x128],
    }

    test_batch_size = {
        "image-mlp-imagenet32-128x128":10000,
        "image-mlp-c10-128x128": 10000,
        "image-mlp-fmst": 10000,
        "image-mlp-fmst64x64": 10000,
        "image-mlp-fmst32x32": 10000,
        "small-image-mlp-fmst": 10000,
        "conv-c10": 10000,
        "small-conv-c10": 10000,
        "conv-imagenet64": 100000,  # TODO Could probably get oom error, fix it when needed
        "conv-imagenet": 100000,
        "fmnist-conv-mlp-mix": 10000,
        "fmnist-mlp-mix": 10000,
        "dataset-mlp-mix": 10000
    }

    batch_size = args.num_grads * args.num_local_steps * args.local_batch_size
    if is_test:
        batch_size = test_batch_size[args.task]

    task = tasks[args.task]

    if type(task) is list:
        return [task(batch_size) for task in task]
    else:
        return tasks[args.task](batch_size)
