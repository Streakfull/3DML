import numpy as np
import os
from pathlib import Path
import torch
import json
from utils.constants import (OVERFIT_DATASET_SIZE,
                             DATASET_PATH,
                             VOX_FOLDERNAME,
                             IMAGE_FOLDERNAME,
                             RENDERINGS_PER_SHAPE,
                             IMAGE_RESOLUTION)
from utils.binvox_rw import read_as_3d_array
from PIL import Image


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13

    class_name_mapping = json.loads(
        Path("datasets/shape_info.json").read_text())
    category_directory_mapping = json.loads(
        Path("datasets/shape_class_info.json").read_text())
    classes = sorted(class_name_mapping.keys())

    def __init__(self, cat="all", is_overfit=True, dataset_path=DATASET_PATH,
                 vox_foldername=VOX_FOLDERNAME, image_foldername=IMAGE_FOLDERNAME):
        super().__init__()
        self.cat = cat
        self.is_overfit = is_overfit
        self.dataset_path = Path(dataset_path)
        self.image_foldername = image_foldername
        self.voxfoldername = vox_foldername
        self.rendering_nimages = RENDERINGS_PER_SHAPE
        self.image_indices = np.arange(self.rendering_nimages)
        self.items = self.get_items()

    def __len__(self):
        if (self.is_overfit and OVERFIT_DATASET_SIZE<len(self.items)):
            return OVERFIT_DATASET_SIZE
        return len(self.items)

    def __getitem__(self, index):
        shape_key = self.items[index]
        voxels = self.get_shape_voxels(shape_key)
        images = self.get_shape_rendering_images(shape_key)
        return {
            "voxels": voxels[np.newaxis, :, :, :],
            "images": images[np.newaxis, :, :, :]
        }

    def get_items(self):
        items = []
        if (self.is_all_categories()):
            for category in self.classes:
                shape_ids = self.get_category_shape_ids(category)
                items.extend(shape_ids)
            return items
        category_id = ShapeNet.category_directory_mapping[self.cat]
        return self.get_category_shape_ids(category_id)

    def get_shape_rendering_images(self, shapenet_key):
        subset_images = np.random.choice(
            self.image_indices, size=self.rendering_nimages, replace=False)
        images = None
        for i, image_number in enumerate(subset_images):
            image_key = f"0{image_number}" if image_number < 10 else image_number
            path = self.dataset_path / self.image_foldername / \
                shapenet_key / "rendering" / f"{image_key}.png"
            image = Image.open(path)
            image_array = np.array(image)[np.newaxis, :, :, :]
            images = image_array if images is None else np.vstack(
                (images, image_array))
        return images

    def get_category_shape_ids(self, category_id):
        ids = os.listdir(
            self.dataset_path / self.voxfoldername / category_id)
        id_categories = map(lambda id: f"{category_id}/{id}", ids)
        return list(id_categories)

    def is_all_categories(self):
        return self.cat == "all"

    def get_shape_voxels(self, shapenet_key):
        with open(self.dataset_path / self.voxfoldername / shapenet_key / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels

    def set_rendering_size(self, rendering_nimages):
        self.rendering_nimages = rendering_nimages
        self.image_indices = np.arange(rendering_nimages)

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['images'] = batch['images'].to(device)
        batch['voxels'] =  batch['voxels'].to(device)
