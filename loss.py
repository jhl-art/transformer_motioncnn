import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import ABC

import logging

logging.basicConfig(filename='baseline_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Loss(ABC, nn.Module):
    def _precision_matrix(shape, sigma_xx, sigma_xy, sigma_yx, sigma_yy):
        assert sigma_xx.shape[-1] == 1
        assert sigma_xx.shape == sigma_yy.shape
        batch_size, n_modes, n_future_timstamps = \
            sigma_xx.shape[0], sigma_xx.shape[1], sigma_xx.shape[2]
        
        # sigma_xx, sigma_xy, sigma_yx, sigma_yy must be positive.
        sigma_xx_inv = 1 / (sigma_xx)
        sigma_xy_inv = 1 / (sigma_xy)
        sigma_yx_inv = 1 / (sigma_yx)
        sigma_yy_inv = 1 / (sigma_yy)

        return torch.cat(
            [sigma_xx_inv, sigma_xy_inv, sigma_yx_inv, sigma_yy_inv], dim=-1) \
            .reshape(batch_size, n_modes, n_future_timstamps, 2, 2)

    def _log_N_conf(self, data_dict, prediction_dict):
        gt = data_dict['gt_path'].unsqueeze(1)
        diff = gt - prediction_dict['xy'].to(torch.float64)

        inf_mask = torch.isinf(diff)
        nan_mask = torch.isnan(diff)
        if inf_mask.any():
            logging.info("diff contains infinity values.")
        if nan_mask.any():
            logging.info("diff contains NaN values.")

        #set diff to nan 
        diff[torch.isnan(diff)] = 0 # mean 
        assert torch.isfinite(diff).all()

        precision_matrices = self._precision_matrix(
            prediction_dict['sigma_xx'],
            prediction_dict['sigma_xy'],
            prediction_dict['sigma_yx'], 
            prediction_dict['sigma_yy'])
        precision_matrices = precision_matrices.to(torch.float64)

        inf_mask = torch.isinf(precision_matrices)
        nan_mask = torch.isnan(precision_matrices)
        if inf_mask.any():
            logging.info("precision contains infinity values.")
        if nan_mask.any():
            logging.info("precision contains NaN values.")

        #set precision to nan
        precision_matrices[torch.isnan(precision_matrices)] = 0
        assert torch.isfinite(precision_matrices).all()


        log_confidences = torch.log_softmax(
            prediction_dict['confidences'], dim=-1)
        assert torch.isfinite(log_confidences).all()

        bilinear = torch.matmul(torch.matmul(diff.unsqueeze(-2), precision_matrices), diff.unsqueeze(-1))
        bilinear = bilinear[:, :, :, 0, 0]
        #bilinear = torch.clamp(bilinear, min=0)
        assert torch.isfinite(bilinear).all()

        conv_matrix = torch.clamp(prediction_dict['sigma_xx'] * prediction_dict['sigma_yy'] - prediction_dict['sigma_yx'] * prediction_dict['sigma_xy'], min=1e-6, max=1e6) 
        log_conv_matrix = torch.log(conv_matrix).squeeze(-1)  # (between -13, 13)
        log_N = -0.5 * np.log(2 * np.pi) - 0.5 * log_conv_matrix - 0.5 * bilinear # between (0, 1)
        return log_N, log_confidences


class NLLGaussian2d(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict, prediction_dict):
        log_N, log_confidences = self._log_N_conf(data_dict, prediction_dict)
        inf_mask = torch.isinf(log_N)
        nan_mask = torch.isnan(log_N)
        if inf_mask.any():
            logging.info("logN contains infinity values.")
        if nan_mask.any():
            logging.info("logN contains NaN values.")
        log_N[torch.isnan(log_N)] = 0
        assert torch.isfinite(log_N).all()
        adjust_log_N = log_N.sum(dim=2) + log_confidences
        log_L = torch.logsumexp(adjust_log_N, dim=1)
        assert torch.isfinite(log_L).all()
        return -log_L.mean()