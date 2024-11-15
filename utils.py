import os
import yaml
import tensorflow as tf
import argparse
from glob import glob

def get_config(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

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
    args = parser.parse_args()
    return args

def get_last_checkpoint_file(path):
    list_of_files = glob(f'{path}/*.pth')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file