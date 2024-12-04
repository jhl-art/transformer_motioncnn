from matplotlib import cm
import numpy as np
from data_sampler import DataSampler
from data_generator import DataGenerator
import os
import tensorflow as tf
import multiprocessing as mp

input_folder = "/mnt/data/validation"
output_folder = "/mnt/data/validation_pkl_advanced"

def generate_data(chunk_id, tfrecord_name):
    input_tfrecord = os.path.join(input_folder, tfrecord_name)
    output_tfrecord_folder = os.path.join(output_folder, tfrecord_name)
    os.makedirs(output_tfrecord_folder, exist_ok=True)

    dataset = tf.data.TFRecordDataset(input_tfrecord, compression_type='')
    iterator = dataset.as_numpy_iterator()
    try:
        segments = [next(iterator) for _ in range(50)]
    except tf.errors.DataLossError as e:
        print("DataLossError error:", e)
    for j in range(50):
        data_sampler = DataSampler(output_tfrecord_folder, j)
        try:
            parsed_feature = data_sampler.parse_feature(segments[j], data_generator.get_feature_description())
        except tf.errors.DataLossError as e:
            print("DataLossError encountered:", e)
            continue
        data_sampler.generate_input_data()

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    data_generator = DataGenerator(input_folder)
    input_record = data_generator.get_input_files()
    num_chunk = len(input_record)
    args = [(i, input_record[i]) for i in range(num_chunk)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(generate_data, args)


