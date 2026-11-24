import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    """
    Optimized dataset class for forest semantic segmentation
    Supports 4-class segmentation: background, living trees, partially dead trees, dead trees
    """

    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, data_augmentation=True):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.data_augmentation = data_augmentation

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # Read images from files
        try:
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        except Exception as e:
            print(f"Failed to read file: {name}, error: {e}")
            return self.__getitem__((index + 1) % self.length)


        jpg, png = self.get_forest_augmented_data(jpg, png, self.input_shape,
                                                  random=self.train,
                                                  use_augmentation=self.data_augmentation)

        # Image preprocessing
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)

        # Label processing - ensure classes are in 0-3 range
        # 0: background, 1: living trees, 2: partially dead trees, 3: dead trees
        png[png >= self.num_classes] = 0

        # Convert to one-hot format
        seg_labels = np.eye(self.num_classes)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_forest_augmented_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4,
                                  random=True, use_augmentation=True):
        pass


        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape


        if not random or not use_augmentation:
            return self._simple_resize(image, label, w, h)


        # 1. Geometric transformations
        image, label = self._apply_geometric_augmentation(image, label, w, h, jitter)

        # 2. Color enhancement - important for forest scenes
        image = self._apply_forest_color_augmentation(image, hue, sat, val)

        # 3. Forest-specific enhancement
        if self.rand() < 0.3:  # 30% probability
            image = self._apply_forest_specific_augmentation(image)

        return image, label

    def _simple_resize(self, image, label, w, h):
        """Simple resize for validation mode or when data augmentation is disabled"""
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', [w, h], (0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

        return new_image, new_label

    def _apply_geometric_augmentation(self, image, label, w, h, jitter):
        """Apply geometric transformation augmentation"""
        iw, ih = image.size

        # Scale and distort image dimensions
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 1.5)  # Suitable scale range for forest data

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # Flip strategy for forest scenes
        # Horizontal flip is safe for forest scenes
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotation augmentation - small angle rotation effective for forest
        if self.rand() < 0.3:  # 30% probability for small angle rotation
            angle = self.rand(-15, 15)  # ±15 degrees
            image = image.rotate(angle, fillcolor=(128, 128, 128))
            label = label.rotate(angle, fillcolor=0)

        # Add gray padding to excess parts
        dx = int(self.rand(0, w - nw)) if w > nw else 0
        dy = int(self.rand(0, h - nh)) if h > nh else 0

        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))

        return new_image, new_label

    def _apply_forest_color_augmentation(self, image, hue, sat, val):
        """
        Color enhancement specific to forest scenes
        Focus on enhancing green channels and contrast to help distinguish tree health states
        """
        image_data = np.array(image, np.uint8)

        # Forest-specific color space transformation
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        # For forest data, appropriately enhance green and brown contrast
        if self.rand() < 0.5:
            # Enhance green channel - helps identify healthy trees
            r[1] *= 1.2  # Slightly increase saturation

        # HSV color space transformation
        try:
            hue_ch, sat_ch, val_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype = image_data.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge((cv2.LUT(hue_ch, lut_hue), cv2.LUT(sat_ch, lut_sat), cv2.LUT(val_ch, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        except:
            # If HSV transformation fails, keep original
            pass

        return Image.fromarray(image_data)

    def _apply_forest_specific_augmentation(self, image):
        """
        Forest-specific augmentation strategies
        Simulate different lighting conditions, seasonal changes, etc.
        """
        augmentation_type = random.choice(['brightness', 'contrast', 'blur', 'noise'])

        if augmentation_type == 'brightness':
            # Simulate lighting at different times
            enhancer = ImageEnhance.Brightness(image)
            factor = self.rand(0.7, 1.3)
            image = enhancer.enhance(factor)

        elif augmentation_type == 'contrast':
            # Enhance contrast to help distinguish tree states
            enhancer = ImageEnhance.Contrast(image)
            factor = self.rand(0.8, 1.4)
            image = enhancer.enhance(factor)

        elif augmentation_type == 'blur':
            # Slight blur to simulate atmospheric conditions
            if self.rand() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=self.rand(0.5, 1.5)))

        elif augmentation_type == 'noise':
            # Add noise to simulate sensor noise
            image_array = np.array(image)
            noise = np.random.normal(0, 5, image_array.shape)
            noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_image)

        return image


class ForestSegmentationDataset(UnetDataset):
    """
    Specialized forest segmentation dataset class
    Contains more optimizations for forest scenes
    """

    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, data_augmentation=True):
        super().__init__(annotation_lines, input_shape, num_classes, train, dataset_path, data_augmentation)

        # Forest segmentation specific class mapping
        self.class_names = ['Background', 'Living Trees', 'Partially Dead Trees', 'Dead Trees']
        self.class_colors = [
            [0, 0, 0],  # Background - Black
            [0, 255, 0],  # Living Trees - Green
            [255, 255, 0],  # Partially Dead Trees - Yellow
            [255, 0, 0]  # Dead Trees - Red
        ]

    def get_class_weights(self):
        """
        Calculate class weights suitable for forest segmentation
        Dead trees usually have fewer samples and need higher weights
        """
        return np.array([1.0, 1.2, 1.5, 1.8], dtype=np.float32)


def unet_dataset_collate(batch):
    """
    Improved collate function with error handling
    """
    images = []
    pngs = []
    seg_labels = []

    for item in batch:
        if item is None:
            continue

        img, png, labels = item
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    if len(images) == 0:
        # Return None if no valid samples
        return None, None, None

    try:
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        pngs = torch.from_numpy(np.array(pngs)).long()
        seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
        return images, pngs, seg_labels
    except Exception as e:
        print(f"Collate function error: {e}")
        return None, None, None


def create_forest_dataloader(annotation_lines, input_shape, num_classes, train, dataset_path,
                             batch_size, num_workers=4, shuffle=True, data_augmentation=True):
    pass


    dataset = ForestSegmentationDataset(
        annotation_lines=annotation_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        train=train,
        dataset_path=dataset_path,
        data_augmentation=data_augmentation
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate
    )

    return dataloader