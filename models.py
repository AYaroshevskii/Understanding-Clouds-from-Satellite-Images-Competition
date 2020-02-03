import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np


dict_of_encoders = {'d121' : ['FPN', 'densenet121'],
                    'd169' : ['FPN', 'densenet169'],
                    'b1' : ['Unet', 'efficientnet-b1'],
                    'b2' : ['Unet', 'efficientnet-b2'],
                    'b3' : ['Unet', 'efficientnet-b3'],
                    'b4' : ['Unet', 'efficientnet-b4']}


class Model(nn.Module):
    def __init__(self, enc):
        super().__init__()
        model_name, encoder = dict_of_encoders[enc]
        
        if model_name == 'Unet':
            self.backbone = smp.Unet(encoder, classes=4,
                                      encoder_weights='imagenet', activation='sigmoid')
        
        if model_name == 'FPN':
            self.backbone = smp.FPN(encoder, classes=4,
                                      encoder_weights='imagenet', activation='sigmoid')

    def forward(self, x):
      
        x = self.backbone(x)

        return x
