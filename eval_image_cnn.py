import numpy as np
import torch
torch.cuda.empty_cache()
import timm
import os
import argparse
from glob import glob
from tqdm import tqdm
from utils import get_config, get_last_checkpoint_file
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from image_cnn_net import SequentialMotionCNN
from loss import NLLGaussian2d
from postprocess import PostProcess
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from train import MotionCNNDataset, parse_arguments
import matplotlib.pyplot as plt
import uuid
import torch.nn.functional as F
import matplotlib.animation as animation
from utils import create_figure_and_axes, create_animation, dict_to_cuda
from evaluator import Evaluator
import random
import csv


def analyze_loss_and_top1mse(model, model_config, loss_module, post_processer, evaluator, dataset):
	
	eval_dataset = MotionCNNDataset(dataset)  # file in the filename
	eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
	eval_progress_bar = tqdm(eval_dataloader, total=len(eval_dataloader))

	mse = []
	fde = []
	weighted_mse = []
	weighted_fde = []
	losses = []

	with torch.no_grad():
		for eval_data in eval_progress_bar:
			torch.cuda.empty_cache()
			eval_data_, data_path = dict_to_cuda(eval_data)
			prediction_tensor = model(eval_data_)
			prediction_dict = post_processer.postprocess_predictions(
	                prediction_tensor, model_config)
			loss = loss_module(eval_data_, prediction_dict)
			top1_mse, top1_fde = evaluator.calculate_top1_mse_and_fde(prediction_dict, eval_data_)
			multi_mse, multi_fde = evaluator.calculate_multitrajectory_mse_and_fde(prediction_dict, eval_data_)
			mse.append(top1_mse.item())
			fde.append(top1_fde.item())
			weighted_mse.append(multi_mse.item())
			weighted_fde.append(multi_fde.item())
			losses.append(loss.item())

	evaluator.dump_top1_mse_and_losses(mse, losses)
	evaluator.dump_top1_fde_and_losses(fde, losses)
	evaluator.dump_multi_mse_and_losses(weighted_mse, losses)
	evaluator.dump_multi_fde_and_losses(weighted_fde, losses)

	return


def analyze_top1_distribution(model, model_config, loss_module, post_processer, evaluator, dataset):
	
	eval_dataset = MotionCNNDataset(dataset)  # file in the filename
	eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
	eval_progress_bar = tqdm(eval_dataloader, total=len(eval_dataloader))

	csv_table_2 = []
	csv_table_5 = []
	csv_table_8 = []
	csv_table_max = []

	fde_csv_table_2 = []
	fde_csv_table_5 = []
	fde_csv_table_8 = []
	fde_csv_table_max = []

	multi_mse_csv_table_2 = []
	multi_mse_csv_table_5 = []
	multi_mse_csv_table_8 = []
	multi_mse_csv_table_max = []

	multi_fde_csv_table_2 = []
	multi_fde_csv_table_5 = []
	multi_fde_csv_table_8 = []
	multi_fde_csv_table_max = []

	type_list = ["mse", "fde", "multi_mse", "multi_fde"]

	with torch.no_grad():
		for eval_data in eval_progress_bar:
			torch.cuda.empty_cache()
			eval_data_, data_path = dict_to_cuda(eval_data)
			pickle_filename = eval_data['pickle_filename']
			prediction_tensor = model(eval_data_)
			prediction_dict = post_processer.postprocess_predictions(
	                prediction_tensor, model_config)
			loss = loss_module(eval_data_, prediction_dict)
			top1_mse, top1_fde = evaluator.calculate_top1_mse_and_fde(prediction_dict, eval_data_)
			multi_mse, multi_fde = evaluator.calculate_multitrajectory_mse_and_fde(prediction_dict, eval_data_)

			for type in type_list:

				if type == "mse":
					mse = top1_mse.item()
					entry = (pickle_filename, loss.item(), top1_mse.item())
					if mse <= 2.0:
						csv_table_2.append(entry)
					elif mse <= 5.0:
						csv_table_5.append(entry)
					elif mse <= 8.0:
						csv_table_8.append(entry)
					else:
						csv_table_max.append(entry)

				if type == "fde":
					fde = top1_fde.item()
					entry = (pickle_filename, loss.item(), top1_fde.item())
					if fde <= 2.0:
						fde_csv_table_2.append(entry)
					elif fde <= 5.0:
						fde_csv_table_5.append(entry)
					elif fde <= 8.0:
						fde_csv_table_8.append(entry)
					else:
						fde_csv_table_max.append(entry)

				if type == "multi_mse":
					multi_mse = multi_mse
					entry = (pickle_filename, loss.item(), multi_mse.item())
					if multi_mse <= 2.0:
						multi_mse_csv_table_2.append(entry)
					elif multi_mse <= 5.0:
						multi_mse_csv_table_5.append(entry)
					elif multi_mse <= 8.0:
						multi_mse_csv_table_8.append(entry)
					else:
						multi_mse_csv_table_max.append(entry)

				if type == "multi_fde":
					multi_fde = multi_fde
					entry = (pickle_filename, loss.item(), multi_fde.item())
					if multi_fde <= 2.0:
						multi_fde_csv_table_2.append(entry)
					elif multi_fde <= 5.0:
						multi_fde_csv_table_5.append(entry)
					elif multi_fde <= 8.0:
						multi_fde_csv_table_8.append(entry)
					else:
						multi_fde_csv_table_max.append(entry)

	evaluator.dump_csv_table(csv_table_2, csv_table_5, csv_table_8, csv_table_max, "mse")
	evaluator.dump_csv_table(fde_csv_table_2, fde_csv_table_5, fde_csv_table_8, fde_csv_table_max, "fde")
	evaluator.dump_csv_table(multi_mse_csv_table_2, multi_mse_csv_table_5, multi_mse_csv_table_8, multi_mse_csv_table_max, "multi_mse")
	evaluator.dump_csv_table(multi_fde_csv_table_2, multi_fde_csv_table_5, multi_fde_csv_table_8, multi_fde_csv_table_max, "multi_fde")
	return

def visualize_pkl_inference_result(model, model_config, loss_module, post_processer, evaluator, dataset):
	eval_dataset = MotionCNNDataset(dataset)  # file in the filename
	eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
	eval_progress_bar = tqdm(eval_dataloader, total=len(eval_dataloader))
	with torch.no_grad():
		for eval_data in eval_progress_bar:
			torch.cuda.empty_cache()
			eval_data_, data_path = dict_to_cuda(eval_data)
			prediction_tensor = model(eval_data_)
			prediction_dict = post_processer.postprocess_predictions(
	                prediction_tensor, model_config)
			loss = loss_module(eval_data_, prediction_dict)
			save_image = evaluator.visualize_result(prediction_dict, eval_data)
			print("Successfully saved image {}", save_image)


def main(checkpoint_path, eval_dataset, evaluator_result_path):
	
	os.makedirs(evaluator_result_path, exist_ok=True)

	general_config = get_config("config/baseline.yaml")
	model_config = general_config['model']
	model = SequentialMotionCNN(model_config).to('cuda')

	loss_module = NLLGaussian2d().to('cuda')
	post_processer = PostProcess(model_config)
	evaluator = Evaluator(evaluator_result_path)

	if checkpoint_path is not None:
		print(f"Loading checkpoint from {checkpoint_path}")
		checkpoint_data = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint_data['model_state_dict'])
		epochs_processed = checkpoint_data['epochs_processed']
	else:
		print(f"No checkpoint found from {checkpoint_path}")
		return

	#analyze_loss_and_top1mse(model, model_config, loss_module, post_processer, evaluator, eval_dataset)
	#analyze_top1_distribution(model, model_config, loss_module, post_processer, evaluator, eval_dataset)
	visualize_pkl_inference_result(model, model_config, loss_module, post_processer, evaluator, eval_dataset)


if __name__ == '__main__':
	checkpoint_path = "/mnt/data/checkpoints/experiment_7_multihead/baseline/e4_b50000.pth"
	eval_dataset = "/mnt/data/test_pkl_e4_b50000/2/"
	evaluation_result_path = "/mnt/data/evaluation/generated_gif_experiment_7_multihead/e4_b50000/gif"
    
	main(checkpoint_path, eval_dataset, evaluation_result_path)

	#visualize_pkl_inference_result()