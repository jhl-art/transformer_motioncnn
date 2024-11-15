from matplotlib import cm
import numpy as np
from data_sampler import DataSampler
from data_generator import DataGenerator
import os
import tensorflow as tf
import multiprocessing as mp


def generate_data(chunk_id, tfrecord_name):
    output_folder = "/mnt/data/training_pkl_raw/" + tfrecord_name
    input_tfrecord = "/mnt/data/training/" + tfrecord_name

    print("input record ", input_tfrecord)
    os.makedirs(output_folder, exist_ok=True)

    dataset = tf.data.TFRecordDataset(input_tfrecord, compression_type='')
    iterator = dataset.as_numpy_iterator()
    try:
        segments = [next(iterator) for _ in range(50)]
    except tf.errors.DataLossError as e:
        print("DataLossError error:", e)
    for j in range(50):
        data_sampler = DataSampler(output_folder, j)
        try:
            parsed_feature = data_sampler.parse_feature(segments[j], data_generator.get_feature_description())
        except tf.errors.DataLossError as e:
            print("DataLossError encountered:", e)
            continue
        data_sampler.generate_training_data(data_generator.roadgraph_color_map,data_generator.agent_color_map, data_generator.traffic_color_map)


if __name__ == "__main__":

    input_folder = "/mnt/data/training"
    output_folder = "/mnt/data/training_pkl_raw"
    os.makedirs(output_folder, exist_ok=True)

    data_generator = DataGenerator(input_folder)
    input_record = data_generator.get_input_files()[:250]
    num_chunk = len(input_record)
    print("num chunk: ", num_chunk)

    args = [(i, input_record[i]) for i in range(num_chunk)]
    print(mp.cpu_count(), " cpu count")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(generate_data, args)

