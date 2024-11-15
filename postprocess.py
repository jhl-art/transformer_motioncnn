import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

# to prevent convariance and standard deviation from negative value 
def limited_softplus(x):
    return torch.clamp(F.softplus(x), min=0.1, max=10)

class PostProcess:
    def __init__(self, model_config):
        self.model_config = model_config
        
    def postprocess_predictions(self, predicted_tensor, model_config):
        confidences = predicted_tensor[:, :model_config['n_modes']]
        components = predicted_tensor[:, model_config['n_modes']:]
        components = components.reshape(
            -1, model_config['n_modes'], model_config['n_timestamps'], 7)
        sigma_xx = components[:, :, :, 2:3]
        sigma_xy = components[:, :, :, 3:4]
        sigma_yx = components[:, :, :, 4:5]
        sigma_yy = components[:, :, :, 5:6]
        visibility = components[:, :, :, 6:]
        return {
            'confidences': confidences,
            'xy': components[:, :, :, :2],
            'sigma_xx': limited_softplus(sigma_xx),
            'sigma_xy': limited_softplus(sigma_xy),
            'sigma_yx': limited_softplus(sigma_yx),
            'sigma_yy': limited_softplus(sigma_yy),
            'visibility': visibility}
        
    