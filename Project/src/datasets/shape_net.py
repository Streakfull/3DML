import numpy as np
import os
from pathlib import Path
import torch
import json
from PIL import Image, ImageFile
from utils.constants import (OVERFIT_DATASET_SIZE,
                             DATASET_PATH,
                             VOX_FOLDERNAME,
                             IMAGE_FOLDERNAME,
                             RENDERINGS_PER_SHAPE,
                             )
from utils.binvox_rw import read_as_3d_array
from PIL import Image
from einops import rearrange
from datasets.transforms import *


ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize(x):
    return x * 2 - 1


def denormalize(x):
    return (x + 1) / 2



def to_numpy(image):
    image.convert("RGB")
    return [np.asarray(image, dtype=np.float32) / 255]

image_trans = Compose([
        to_numpy,
        CenterCrop((224, 224), (128, 128)),
        RandomBackground(((240, 240), (240, 240), (240, 240))),
        lambda x: x[0],
        normalize
    ])

# ImageFile.LOAD_TRUNCATED_IMAGES = True


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13

    class_name_mapping = json.loads(
        Path("datasets/shape_info.json").read_text())
    category_directory_mapping = json.loads(
        Path("datasets/shape_class_info.json").read_text())
    classes = sorted(class_name_mapping.keys())
    class_names = sorted(class_name_mapping.values())

    def __init__(self, cat="all", is_overfit=True, dataset_path=DATASET_PATH,
                 vox_foldername=VOX_FOLDERNAME, image_foldername=IMAGE_FOLDERNAME, nimgs=1):
        super().__init__()
        self.cat = cat
        self.is_overfit = is_overfit
        self.dataset_path = Path(dataset_path)
        self.image_foldername = image_foldername
        self.voxfoldername = vox_foldername
        self.rendering_nimages = RENDERINGS_PER_SHAPE
        self.image_indices = np.arange(self.rendering_nimages)
        self.items = self.get_items()
        self.nimgs = nimgs
        self.background=(255, 255, 255),
        self.image_transforms = image_trans

    def __len__(self):
        if (self.is_overfit and OVERFIT_DATASET_SIZE<len(self.items)):
            return OVERFIT_DATASET_SIZE
        return len(self.items)

    def __getitem__(self, index):
        shape_key = self.items[index]
        voxels = self.get_shape_voxels(shape_key)
        images = self.get_shape_rendering_images(shape_key)
        images = self.transform_images(images)
        raw_images = self.get_raw_images(shape_key)
        shape_info = shape_key.split("/")
        class_name = ShapeNet.class_name_mapping[shape_info[0]]
        
        return {
            "voxels": voxels[np.newaxis, :, :, :],
            "images": images,
            "class": class_name,
            "raw_image": raw_images,
            "id": shape_info[1]
        }
    
    def transform_images(self, images):
        if(self.nimgs == 1):
            images = images[0]
            images = rearrange(images, 'h w c -> c h w')
            images = images[np.newaxis, :, :, :]
        else:
#             images = images[0:self.nimgs]
            images = rearrange(images,'nimgs h w c -> nimgs c h w')
        return images

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
        subset_images = subset_images[0 : self.nimgs]
        images = None
        for i, image_number in enumerate(subset_images):
            image_key = f"0{image_number}" if image_number < 10 else image_number
            path = self.dataset_path / self.image_foldername / \
                shapenet_key / "rendering" / f"{image_key}.png"
            rgba = Image.open(path)
            image = Image.new("RGB", rgba.size, self.background)
            image.paste(rgba, mask=rgba.split()[3])
            image = self.image_transforms(image)
            image_array = image[np.newaxis, :, :, :]
            #image_array = image_array / 255
            images = image_array if images is None else np.vstack(
                (images, image_array))
        return images
    
    def get_raw_images(self, shapenet_key):
         subset_images = np.random.choice(
            self.image_indices, size=1, replace=False)
         image_number = subset_images[0]
         image_key = f"0{image_number}" if image_number < 10 else image_number
         path = self.dataset_path / self.image_foldername / shapenet_key / "rendering" / f"{image_key}.png"
         rgba = Image.open(path)
         rgba = np.array(rgba) / 255
         return rgba

    def get_category_shape_ids(self, category_id):
        ids = os.listdir(
            self.dataset_path / self.voxfoldername / category_id)
        id_categories = map(lambda id: f"{category_id}/{id}", ids)
        return list(id_categories)

    def is_all_categories(self):
        return self.cat == "all"

    def get_shape_voxels(self, shapenet_key):
        #print(shapenet_key,"KEY")
        with open(self.dataset_path / self.voxfoldername / shapenet_key / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels

    def set_rendering_size(self, rendering_nimages):
        self.rendering_nimages = rendering_nimages
        self.image_indices = np.arange(rendering_nimages)
    
    def set_nimgs(self,nimgs):
        self.nimgs = nimgs

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['images'] = batch['images'].float().to(device)
        batch['voxels'] =  batch['voxels'].float().to(device)
        batch["raw_image"] = batch["raw_image"].float().to(device)
       
   
    @staticmethod
    def move_batch_to_device_float(batch, device):
        batch['images'] = batch['images'].float()
        batch['voxels'] =  batch['voxels'].float()
        batch["raw_image"] = batch["raw_image"].float()
        
    
