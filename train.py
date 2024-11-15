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
from torch.optim import Adam
from image_cnn_net import SequentialMotionCNN
from loss import NLLGaussian2d
from postprocess import PostProcess
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/mnt/data/image_cnn/runs/softmax_log")

class MotionCNNDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self._files = glob(f"{data_path}/**/*.pkl", recursive=True)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        data = dict(np.load(self._files[idx], allow_pickle=True))
        return data

def dict_to_cuda(data):
    train_data = {}
    data_path = []
    for key, value in data.items():
        if isinstance(value, torch.Tensor) is False:
            data_path.append(value)
            continue
        train_data[key] = value.to('cuda')
    return train_data, data_path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, required=True,
        help="Path to training data")
    parser.add_argument(
        "--val-data-path", type=str, required=True,
        help="Path to validation data")
    parser.add_argument(
        "--checkpoints-path", type=str, required=True,
        help="Path to checkpoints")
    parser.add_argument(
        "--config", type=str, required=True, help="Config file path")
    parser.add_argument("--multi-gpu", action='store_true')
    parser.add_argument("--event-log-path",type=str, required=True,
        help="Path to training data")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    general_config = get_config(args.config)
    model_config = general_config['model']
    training_config = general_config['training']
    config_name = args.config.split('/')[-1].split('.')[0]
    model = SequentialMotionCNN(model_config).to('cuda')
    optimizer = Adam(model.parameters(), **training_config['optimizer'])
    loss_module = NLLGaussian2d().to('cuda')
    processed_batches = 0
    epochs_processed = 0
    train_losses = []
    print("event log path: ", args.event_log_path)
    writer = SummaryWriter(args.event_log_path)
    experiment_checkpoints_dir = os.path.join(
        args.checkpoints_path, config_name)
    if not os.path.exists(experiment_checkpoints_dir):
        os.makedirs(experiment_checkpoints_dir)
    latest_checkpoint = get_last_checkpoint_file(experiment_checkpoints_dir)
    print(latest_checkpoint)
    if latest_checkpoint is not None:
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint_data = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        #optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        epochs_processed = checkpoint_data['epochs_processed']
        processed_batches = checkpoint_data.get('processed_batches', 0)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    training_dataloader = DataLoader(
        MotionCNNDataset(args.train_data_path),
        **training_config['train_dataloader'])
    validation_dataloader = DataLoader(
        MotionCNNDataset(args.val_data_path),
        **training_config['val_dataloader'])
    post_processer = PostProcess(model_config)

    for epochs_processed in tqdm(
            range(epochs_processed, training_config['num_epochs']),
            total=training_config['num_epochs'],
            initial=epochs_processed):
        running_loss = 0.0
        train_progress_bar = tqdm(
            training_dataloader, total=len(training_dataloader))
        for train_data in train_progress_bar:
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            train_data, data_path = dict_to_cuda(train_data)
            model_device = next(model.parameters()).device
            prediction_tensor = model(train_data)
            prediction_dict = post_processer.postprocess_predictions(
                prediction_tensor, model_config)
            loss = loss_module(train_data, prediction_dict)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            running_loss += loss.item()
            processed_batches += 1
            train_progress_bar.set_description(
                "Train loss: %.3f" % np.mean(train_losses[-100:]))
            if processed_batches % training_config['eval_every'] == 0:
                del train_data
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for eval_data in tqdm(validation_dataloader):
                        eval_data,data_path = dict_to_cuda(eval_data)
                        prediction_tensor = model(eval_data)
                        prediction_dict = \
                             post_processer.postprocess_predictions(
                            prediction_tensor, model_config)
                        loss = loss_module(eval_data, prediction_dict)
                if  isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch_checkpoint_data = {
                    "model_state_dict": model_state_dict,
                    "epochs_processed": epochs_processed}
                torch_checkpoint_path = os.path.join(
                    experiment_checkpoints_dir,
                    f'e{epochs_processed}_b{processed_batches}.pth')
                torch.save(torch_checkpoint_data, torch_checkpoint_path)
        avg_loss = running_loss / len(training_dataloader)
        writer.add_scalar("Training Loss", avg_loss, epochs_processed)



if __name__ == '__main__':
    main()