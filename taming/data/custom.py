import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from torchvision import transforms as T
import h5py

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class TextImageDatasetFashionGen(Dataset):
    def __init__(self, h5filepath, image_size = 256, split="train"):
        super().__init__()
        self.file_h5 = h5py.File(h5filepath, mode='r')
        
        if split == "train":
          self.image_transform = T.Compose([
              T.ToPILImage(),
              T.RandomResizedCrop(image_size, scale = (0.75, 1.), ratio = (1., 1.)),
              T.RandomHorizontalFlip(p=0.3),
              T.ColorJitter(brightness=0.025, hue=0.0, contrast=0.025, saturation=0.2),
          ])
        else:
          self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
          ])

    def __len__(self):
        return len(self.file_h5['input_image'])

    def __getitem__(self, ind):
        data = dict()
        image = self.file_h5['input_image'][ind]
        image = self.image_transform(image)
        image = (np.array(image)/127.5 - 1.0).astype(np.float32)
        data["image"] = image

        return data