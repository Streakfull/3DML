import os
from termcolor import colored, cprint
import torch
import utils.util as util

# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.is_train= opt.is_train

        if self.is_train:
            self.save_dir = os.path.join(opt.logs_dir, opt.name, 'checkpoints')

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                cprint(f"{self.save_dir} created", "blue")

        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def backward(self):
        pass

    def step(self):
        pass

    def get_image_paths(self):
        pass

    def save(self, label):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass

    def inference(self):
        pass

    def get_current_errors(self):
        return 0
    
    def update_lr(self):
        pass


    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(0, non_blocking=True))
