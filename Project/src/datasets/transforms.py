#adapted from https://github.com/fomalhautb/3D-RETR

import os
import random

import cv2
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images = t(rendering_images, bounding_box)
            else:
                rendering_images = t(rendering_images)

        return rendering_images

class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, rendering_images, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images

        crop_size_c = rendering_images[0].shape[2]
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if bounding_box is not None:
                bounding_box = [
                    bounding_box[0] * img_width,
                    bounding_box[1] * img_height,
                    bounding_box[2] * img_width,
                    bounding_box[3] * img_height
                ]  # yapf: disable

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

                # Make the crop area as a square
                square_object_size = max(bbox_width, bbox_height)
                x_left = int(bbox_x_mid - square_object_size * .5)
                x_right = int(bbox_x_mid + square_object_size * .5)
                y_top = int(bbox_y_mid - square_object_size * .5)
                y_bottom = int(bbox_y_mid + square_object_size * .5)

                # If the crop position is out of the image, fix it with padding
                pad_x_left = 0
                if x_left < 0:
                    pad_x_left = -x_left
                    x_left = 0
                pad_x_right = 0
                if x_right >= img_width:
                    pad_x_right = x_right - img_width + 1
                    x_right = img_width - 1
                pad_y_top = 0
                if y_top < 0:
                    pad_y_top = -y_top
                    y_top = 0
                pad_y_bottom = 0
                if y_bottom >= img_height:
                    pad_y_bottom = y_bottom - img_height + 1
                    y_bottom = img_height - 1

                # Padding the image and resize the image
                processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                         mode='edge')
                processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = int(img_width - self.crop_size_w) // 2
                    x_right = int(x_left + self.crop_size_w)
                    y_top = int(img_height - self.crop_size_h) // 2
                    y_bottom = int(y_top + self.crop_size_h)
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

                processed_image = cv2.resize(img[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

            processed_images = np.append(processed_images, [processed_image], axis=0)
            # Debug
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(img)
            # if not bounding_box is None:
            #     rect = patches.Rectangle((bounding_box[0], bounding_box[1]),
            #                              bbox_width,
            #                              bbox_height,
            #                              linewidth=1,
            #                              edgecolor='r',
            #                              facecolor='none')
            #     ax1.add_patch(rect)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images

class RandomBackground(object):
    def __init__(self, random_bg_color_range, random_bg_folder_path=None):
        self.random_bg_color_range = random_bg_color_range
        self.random_bg_files = []
        if random_bg_folder_path is not None:
            self.random_bg_files = os.listdir(random_bg_folder_path)
            self.random_bg_files = [os.path.join(random_bg_folder_path, rbf) for rbf in self.random_bg_files]

    def __call__(self, rendering_images):
        if len(rendering_images) == 0:
            return rendering_images

        img_height, img_width, img_channels = rendering_images[0].shape
        # If the image has the alpha channel, add the background
        if not img_channels == 4:
            return rendering_images

        # Generate random background
        r, g, b = np.array([
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]) / 255.

        random_bg = None
        if len(self.random_bg_files) > 0:
            random_bg_file_path = random.choice(self.random_bg_files)
            random_bg = cv2.imread(random_bg_file_path).astype(np.float32) / 255.

        # Apply random background
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels - 1))
        for img_idx, img in enumerate(rendering_images):
            alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
            img = img[:, :, :3]
            bg_color = random_bg if random.randint(0, 1) and random_bg is not None else np.array([[[r, g, b]]])
            img = alpha * bg_color + (1 - alpha) * img

            processed_images = np.append(processed_images, [img], axis=0)

        return processed_images
