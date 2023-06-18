import numpy as np
from pathlib import Path
import torch
import json
from utils.constants import OVERFIT_DATASET_SIZE

class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13
    class_name_mapping = json.loads(Path("/datasets/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, cat="all", is_overfit = True):
        super().__init__()
        self.cat = cat
        self.is_overfit = is_overfit
        self.items = self.set_items()


    def __len__(self):
        if(self.is_overfit):
            return OVERFIT_DATASET_SIZE
        return len(self.items)
    


    def __getitem__(self,index):
        print("HI")
        

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['images'] = batch['images'].to(device)
        batch['target_df'] = batch['target_df'].to(device)
        pass