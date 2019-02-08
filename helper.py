import math
import os
import hashlib
import shutil

import numpy as np
from PIL import Image

def get_image(image_path, mode,
             preprocessing_function=lambda x, **kw: x,
             **preprocessing_kwargs):
    return np.array(preprocessing_function(Image.open(image_path), **preprocessing_kwargs).convert(mode))

def celeba_preprocessing(image, width, height, **kwargs):
    # Remove most pixels that aren't part of a face
    # taken from medium.com
    face_width = face_height = kwargs.get('face_size', 108)
    j = (image.size[0] - face_width) // 2 + kwargs.get('delta_j', 0)
    i = (image.size[1] - face_height) // 2 + kwargs.get('delta_i', 0)
    return image.rotate(kwargs.get('rotation', 0)).crop([j, i, j + face_width, i + face_height]).resize([width, height], Image.BILINEAR)

def load(image_files, mode, preprocessing_function=lambda x, **kw: x, **preprocessing_kwargs):
    data_batch = np.array(
        [get_image(sample_file, mode,
                   preprocessing_function,
                   **preprocessing_kwargs) for sample_file in image_files]).astype(np.float32)

    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch / 255 - 0.5

def image_for_plot(i):
    return (((i - i.min()) * 255) / (i.max() - i.min())).astype(np.uint8)


def images_grid_for_plot(images, mode, grid_width, grid_height):
    images = np.array(images).reshape(grid_width, grid_height, *images[0].shape)
    return np.vstack([np.hstack([images[j, i] for j in range(grid_width)]) for i in range(grid_height)])


class ImagesDataset:
    __slots__ = ['images', 'loaded_images', 'preprocessing_function', 'preprocessing_kwargs', 'image_mode', 'shape']
    def __init__(self, images_dirs, image_mode, preprocessing_function, **preprocessing_kwargs):
        self.images = load(images_dirs, image_mode, preprocessing_function, **preprocessing_kwargs)
        self.preprocessing_function = preprocessing_function
        self.preprocessing_kwargs = preprocessing_kwargs
        self.image_mode = image_mode
        self.shape = len(images_dirs), preprocessing_kwargs['width'], preprocessing_kwargs['height'], {'RGB': 3, 'L': 1}[image_mode]

    def get_batches(self, batch_size):
        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            yield self.images[current_index:current_index + batch_size, :, :, :]
            current_index += batch_size

    def get_random_batch(self, batch_size):
        return self.images[np.random.randint(0, self.shape[0], size=batch_size), :, :, :]

# class ImagesDataset:
#     def __init__(self, images_dirs, image_mode, preprocessing_function, **preprocessing_kwargs):
#         self.images_dirs = images_dirs
#         self.preprocessing_function = preprocessing_function
#         self.preprocessing_kwargs = preprocessing_kwargs
#         self.image_mode = image_mode
#         self.shape = len(images_dirs), preprocessing_kwargs['width'], preprocessing_kwargs['height'], {'RGB': 3, 'L': 1}[image_mode]
#
#     def get_batches(self, batch_size):
#         current_index = 0
#         while current_index + batch_size <= self.shape[0]:
#             yield load(
#                 self.images_dirs[current_index:current_index + batch_size],
#                 self.image_mode,
#                 self.preprocessing_function,
#                 **self.preprocessing_kwargs
#                 )
#
#             current_index += batch_size
#
#     def get_random_batch(self, batch_size):
#         random_images = np.array(self.images_dirs)[np.random.randint(0, self.shape[0], size=batch_size)]
#         return load(list(random_images), self.image_mode, self.preprocessing_function, **self.preprocessing_kwargs)
