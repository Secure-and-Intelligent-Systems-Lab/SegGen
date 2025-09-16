import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any
import cv2
import random
from components.factory.factory import DATASETS

@DATASETS.register("UE5")
class UE5(Dataset):
    """
    num_classes: 6
    """
    CLASSES = ['Ground', 'European Beech', 'Rocks', 'Norway Maple', 'Dead', 'Black Alder']

    PALETTE = torch.tensor([[112, 56, 0],
                            [0, 137, 0],
                            [182, 201, 206],
                            [105, 78, 146],
                            [135, 25, 13],
                            [200, 15, 15]])


    def __init__(self, root: str = 'data/UE5Dataset/', split: str = 'train', transform = None, modals=None) -> None:
        super().__init__()
        assert [modal in ['image', 'depth'] for modal in modals]
        assert split in ['train', 'val']
        self.split = split
        self.root = root
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self._left_offset = 0
        self.img_h = 1080
        self.img_w = 1920
        self.base_size = 1080
        self.crop_size = (1280, 768)
        self.test_size = (1792, 1024)
        self.files = self._get_file_names(split)

        if not self.files:
            raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[list[Any], Any]:
        item_name = str(self.files[index])
        rgb = os.path.join(*[self.root, 'Optical', item_name])
        x1 = os.path.join(*[self.root, 'Depth', item_name])
        lbl_path = os.path.join(*[self.root, 'Labels_Processed', item_name])
        _img = cv2.imread(rgb, -1)[..., :3]
        _img = _img[:, :, ::-1]
        _img = _img.astype(np.float32) / 65535 if _img.dtype == np.uint16 else _img.astype(np.float32) / 255
        _target = cv2.imread(lbl_path, -1)
        _target = _target.astype(np.float32)
        _depth = cv2.imread(x1, -1)[..., :3]
        _depth = _depth.astype(np.float32) / 65535 if _depth.dtype == np.uint16 else _depth.astype(np.float32) / 255

        sample = {'image': _img, 'label': _target, 'depth': _depth}

        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        else:
            raise NotImplementedError()
        label = sample['label'].long()
        sample = [sample[k] for k in self.modals]
        # del _img, _target, _depth
        return sample, label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=self.test_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        list_path = os.path.join(self.root, f"{'train' if split_name == 'train' else 'val'}.txt")

        with open(list_path, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]

        return file_names


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img -= self.mean
        img /= self.std

        depth = sample['depth']
        depth = np.array(depth).astype(np.float32)
        depth -= self.mean
        depth /= self.std

        return {'image': img,
                'label': mask,
                'depth': depth,
                }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        depth = np.array(depth).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        depth = torch.from_numpy(depth).float()

        return {'image': img,
                'label': mask,
                'depth': depth}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']
        if random.random() < 0.5:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # nir = nir.transpose(Image.FLIP_LEFT_RIGHT)

            img = img[:, ::-1]
            mask = mask[:, ::-1]
            depth = depth[:, ::-1]

        return {'image': img,
                'label': mask,
                'depth': depth,
                }


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']
        if random.random() < 0.5:
            radius = random.random()
            img = cv2.GaussianBlur(img, (0, 0), radius)
            depth = cv2.GaussianBlur(depth, (0, 0), radius)

        return {'image': img,
                'label': mask,
                'depth': depth,
                }


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        h, w = img.shape[:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # pad crop
        if short_size < self.crop_size[1]:
            padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0
            padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0

        x1 = random.randint(0, max(0, ow - self.crop_size[0]))
        y1 = random.randint(0, max(0, oh - self.crop_size[1]))

        depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if short_size < self.crop_size[1]:
            depth_ = np.zeros((oh + padh, ow + padw, 3))
            depth_[:oh, :ow] = depth
            depth = depth_

            img_ = np.zeros((oh + padh, ow + padw, 3))
            img_[:oh, :ow] = img
            img = img_
            mask_ = np.full((oh + padh, ow + padw), self.fill)
            mask_[:oh, :ow] = mask
            mask = mask_

        depth = depth[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]
        img = img[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]
        mask = mask[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]
        return {'image': img,
                'label': mask,
                'depth': depth,
                }


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        # w, h = img.size
        h, w = img.shape[:2]

        if w > h:
            oh = self.crop_size[1]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[0]
            oh = int(1.0 * h * ow / w)

        x1 = int(round((ow - self.crop_size[0]) / 2.))
        y1 = int(round((oh - self.crop_size[1]) / 2.))

        depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        depth = depth[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]
        img = img[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]
        mask = mask[y1:y1 + self.crop_size[1], x1:x1 + self.crop_size[0]]

        return {'image': img,
                'label': mask,
                'depth': depth,
                }
