from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')
  
        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        # TODO Stack (distances, sdf sign) for the input sdf
        # TODO Log-scale target df
        input_sdf = np.clip(input_sdf,-3,3)
        target_df = np.clip(target_df,-3,3)
        input_sdf_signed = np.abs(input_sdf)
        signed_input_sdf = np.ones((32,32,32))
        negative_indices = input_sdf < 0
        signed_input_sdf[negative_indices] = -1
        input_sdf = np.vstack((input_sdf_signed[np.newaxis,:,:,:],signed_input_sdf[np.newaxis,:,:,:]))
        target_df = np.log(target_df + 1)
        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['input_sdf'].to(device)
        batch['target_df'].to(device)
        pass

    @staticmethod
    def get_shape_sdf(shapenet_id):
        item_class = shapenet_id.split("/")[0]
        item_id = shapenet_id.split("/")[1]
        dt = np.dtype([('dimX',np.uint64), ('dimY',np.uint64),('dimZ',np.uint64)])
        dimensions = np.fromfile(f"{ShapeNet.dataset_sdf_path}/{item_class}/{item_id}.sdf", dtype=np.uint64, count =3)
        sdf = np.fromfile(f"{ShapeNet.dataset_sdf_path}/{item_class}/{item_id}.sdf",dtype=np.float32, offset=24)
        sdf = np.reshape(sdf,dimensions)
        # TODO implement sdf data loading
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        item_class = shapenet_id.split("/")[0]
        item_id = shapenet_id.split("/")[1]
        dt = np.dtype([('dimX',np.uint64), ('dimY',np.uint64),('dimZ',np.uint64)])
        dimensions = np.fromfile(f"{ShapeNet.dataset_df_path}/{item_class}/{item_id}.df", dtype=np.uint64, count =3)
        df = np.fromfile(f"{ShapeNet.dataset_df_path}/{item_class}/{item_id}.df",dtype=np.float32,offset=24)
        df = np.reshape(df,dimensions)
        # TODO implement df data loading
        return df
