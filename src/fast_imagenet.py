import h5py
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import smart_open

class ImageNetDatasetH5(Dataset):
    def __init__(self, root, split, transform=None, albumentations=False, **kwargs):
        print("unused kwargs:", kwargs)
        self.h5_path = root  # Path to ilsvrc2012.hdf5
        self.root = root
        self.split = split
        self.transform = transform
        self.albumentations = albumentations
        assert os.path.exists(self.h5_path), f"ImageNet h5 file path does not exist! Given: {self.h5_path}"
        assert self.split in ["train", "val", "test"], f"split must be 'train' or 'val' or 'test'! Given: {self.split}"
        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = 100000
        self.h5_data = None

    def __len__(self):
        if self.split == "train":
            return self.n_train
        elif self.split == "val":
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, idx):
        # Correct idx
        if self.split == 'val':
            idx += self.n_train
        elif self.split == 'test':
            idx += self.n_train + self.n_val
        # Read h5 file
        if self.h5_data is None:

            self.h5_data = h5py.File(smart_open.open(self.h5_path, "rb"), mode='r')
            # print([d for d in self.h5_data])
            # print(self.h5_data['targets'][0])
        # Extract info
        image = Image.open(io.BytesIO(self.h5_data['encoded_images'][idx])).convert('RGB')
        
        if self.transform is not None:
            if self.albumentations:
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        target = self.h5_data['targets'][idx][0] if self.split != 'test' else None
        return image, target
    

class ImageNetDatasetH5Mem(Dataset):
    def __init__(self, root, split, transform=None, albumentations=False, **kwargs):
        print("unused kwargs:", kwargs)
        self.h5_path = root  # Path to ilsvrc2012.hdf5
        self.root = root
        self.split = split
        self.transform = transform
        self.albumentations = albumentations
        assert os.path.exists(self.h5_path), f"ImageNet h5 file path does not exist! Given: {self.h5_path}"
        assert self.split in ["train", "val", "test"], f"split must be 'train' or 'val' or 'test'! Given: {self.split}"
        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = 100000
        self.h5_data = None

        with h5py.File(smart_open.open(self.h5_path, "rb"), mode='r') as h5_file:    
            # Assuming the dataset has 'images' and 'labels' keys
            self.images = h5_file['encoded_images'][:]
            self.labels = h5_file['targets'][:]

    
    def __len__(self):
        if self.split == "train":
            return self.n_train
        elif self.split == "val":
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, idx):
        # Correct idx
        if self.split == 'val':
            idx += self.n_train
        elif self.split == 'test':
            idx += self.n_train + self.n_val
        # Read h5 file
        # if self.h5_data is None:

        #     self.h5_data = h5py.File(smart_open.open(self.h5_path, "rb"), mode='r')
            # print([d for d in self.h5_data])
            # print(self.h5_data['targets'][0])
        # Extract info
        image = Image.open(io.BytesIO(self.images[idx])).convert('RGB')
        
        if self.transform is not None:
            if self.albumentations:
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        target = self.labels[idx][0] if self.split != 'test' else None
        return image, target

class ImageNet21kDatasetH5(Dataset):
    def __init__(self, root, split, transform=None, albumentations=False, n_train=1281167, n_val=50000, **kwargs):
        print("unused kwargs:", kwargs)
        self.h5_path = root  # Path to ilsvrc2012.hdf5
        self.root = root
        self.split = split
        self.transform = transform
        self.albumentations = albumentations
        assert os.path.exists(self.h5_path), f"ImageNet h5 file path does not exist! Given: {self.h5_path}"
        assert self.split in ["train", "val", "test"], f"split must be 'train' or 'val' or 'test'! Given: {self.split}"
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = 100000
        self.h5_data = None

    def __len__(self):
        if self.split == "train":
            return self.n_train
        elif self.split == "val":
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, idx):
        # Correct idx
        if self.split == 'val':
            idx += self.n_train
        elif self.split == 'test':
            idx += self.n_train + self.n_val
        # Read h5 file
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_path, mode='r')
            # print([d for d in self.h5_data])
            # print(self.h5_data['targets'][0])
        # Extract info
        image = Image.open(io.BytesIO(self.h5_data['encoded_images'][idx])).convert('RGB')
        if self.transform is not None:
            if self.albumentations:
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        target = torch.from_numpy(self.h5_data['targets'][idx])[0].long() if self.split != 'test' else None
        return image, target


def ImageNetDataLoaders(h5_path, batch_size, workers=10, distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = ImageNetDatasetH5(h5_path, "train", train_transform)
    val_ds = ImageNetDatasetH5(h5_path, "val", val_transform)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        sampler = None
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)
    return train_dl, val_dl


if __name__ == "__main__":
    h5_path = "/home/datasets/imagenet_hdf5/ilsvrc2012.hdf5"
    dataset = ImageNetDatasetH5(h5_path, "train")
    for x, y in dataset:
        print(x.size,y)




import jax.numpy as jnp
from learned_optimization.tasks.datasets.base import Datasets,ThreadSafeIterator,LazyIterator
import jax
import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None,prefetch_factor=200,):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
  


import torch
from collections import namedtuple

# Define the FlatMap structure
# FlatMap = namedtuple('FlatMap', ['image', 'label'])
import haiku as hk

class DataLoaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            images, labels = batch

            # print(images.shape, labels.shape)
            # Transform the image tensor shape from (batch_size, channels, height, width) 
            # to (1, batch_size, height, width, channels)
            # images = images.permute(0, 2, 3, 1) #.unsqueeze(0)
            # exit(0)

            # Ensure labels have the shape (1, batch_size)
            # labels = labels.unsqueeze(0)

            # Yield the data in the desired FlatMap format
            yield hk.data_structures.to_immutable_dict({"image":images, "label":labels})


def fast_imagenet_datasets(h5_path, batch_size, workers=10, distributed=False, image_size=(32,32,), output_channel=(3,)):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size[0]),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize,
        np.asarray
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size[0]),
        # transforms.CenterCrop(image_size[0]),
        # transforms.ToTensor(),
        # normalize,
        np.asarray
    ])
    train_ds = ImageNetDatasetH5Mem(h5_path, "train", train_transform)
    # val_ds = ImageNetDatasetH5Mem(h5_path, "val", val_transform)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        sampler = None

    train_dl = NumpyLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)
    # val_dl = NumpyLoader(
    #     val_ds, batch_size=batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True, drop_last=True)

    train_dl = iter(DataLoaderWrapper(train_dl))
    # val_dl = iter(DataLoaderWrapper(val_dl))


    abstract_batch = {
        "image":
            jax.core.ShapedArray(
                (batch_size,) + image_size + output_channel, dtype=jnp.float32),
        "label":
            jax.core.ShapedArray((batch_size,), dtype=jnp.int32)
    }

    return Datasets(
               train=train_dl,
               inner_valid=train_dl,
               outer_valid=train_dl,
               test=train_dl,
               extra_info={"num_classes": 1000, 'name':'fastimagenet'},
               abstract_batch=abstract_batch)
               
               