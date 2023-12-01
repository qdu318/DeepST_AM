# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import Z_Score
from utils.utils import generate_dataset
from utils.utils import generate_asist_dataset
from utils.utils import *


def Data_load(config, timesteps_input, timesteps_output, num_of_weeks, num_of_days, num_of_hours, ponits_per_hour = 12):
    # W_nodes = pd.read_csv(config['W_nodes'], header=None).to_numpy(np.float32)
    W_nodes = np.load(config['NATree']).astype(np.float32)
    X = pd.read_csv(config['V_nodes'], header=None).to_numpy(np.float32) * 100
    V_confirmed = pd.read_csv(config['V_confirmed'], header=None).to_numpy(np.float32)
    V_cured = pd.read_csv(config['V_cured'], header=None).to_numpy(np.float32)
    V_suspected = pd.read_csv(config['V_suspected'], header=None).to_numpy(np.float32)
    V_dead = pd.read_csv(config['V_dead'], header=None).to_numpy(np.float32)
    Weather = np.load(config['Weather']).astype(np.float32)
    Weather=Weather[0:96]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    X, X_mean, X_std = Z_Score(X)

    V_confirmed = np.reshape(V_confirmed, (V_confirmed.shape[0], V_confirmed.shape[1], 1)).transpose((1, 2, 0))
    V_confirmed, _, _ = Z_Score(V_confirmed)
    V_cured = np.reshape(V_cured, (V_cured.shape[0], V_cured.shape[1], 1)).transpose((1, 2, 0))
    V_cured, _, _ = Z_Score(V_cured)
    V_suspected = np.reshape(V_suspected, (V_suspected.shape[0], V_suspected.shape[1], 1)).transpose((1, 2, 0))
    V_suspected, _, _ = Z_Score(V_suspected)
    V_dead = np.reshape(V_dead, (V_dead.shape[0], V_dead.shape[1], 1)).transpose((1, 2, 0))
    V_dead, _, _ = Z_Score(V_dead)
    # V_conbine = np.concatenate((X, V_confirmed, V_cured, V_suspected, V_dead), axis=1)
    Weather, _, _ = Z_Score(Weather)
    V_conbine = np.concatenate((X, V_confirmed, V_cured, V_suspected, V_dead), axis=1)
    V_conbine = V_conbine.transpose((2, 0, 1))

    all_samples = []
    for idx in range(V_conbine.shape[0]):
        sample = get_sample_indices(data_sequence=V_conbine, num_of_weeks=num_of_weeks, num_of_days=num_of_days,
                                    num_of_hours=num_of_hours, label_start_idx=idx, num_for_predict=timesteps_output,
                                    points_per_hour=timesteps_input)

        if not sample:
            continue
        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))
    # index_1 = int(X.shape[2] * 0.8)
    # index_2 = int(X.shape[2])
    # train_original_data = X[:, :, :index_2]
    # train_asist = V_conbine[:, :, :index_2]
    # val_original_data = X[:, :, :index_2]
    # val_asist = V_conbine[:, :, :index_2]
    # index_1 = int(Weather.shape[0] * 0.8)
    # index_2 = int(Weather.shape[0])
    # train_weather = Weather[:index_1]
    # val_weather = Weather[index_1:index_2]

    index_1 = int(len(all_samples)*0.6)
    index_2 = int(len(all_samples)*0.8)
    # merget train_set and eval_set
    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:index_2])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[index_1: index_2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:index_2])]
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set
    train_weather = Weather[:index_1]
    val_weather = Weather[index_1:index_2]

    # train_input, train_target = generate_dataset(train_original_data,
    #                                              num_timesteps_input=timesteps_input,
    #                                              num_timesteps_output=timesteps_output)
    # evaluate_input, evaluate_target = generate_dataset(val_original_data,
    #                                                    num_timesteps_input=timesteps_input,
    #                                                    num_timesteps_output=timesteps_output)
    # train_asist_dataset = generate_asist_dataset(train_asist, timesteps_input, timesteps_output)
    # val_asist_dataset = generate_asist_dataset(val_asist, timesteps_input, timesteps_output)

    data_set = {
        'train': {
            'week': train_week,
            'day': train_day,
            'recent': train_hour,
            'target': train_target,
        },
        'val': {
            'week': val_week,
            'day': val_day,
            'recent': val_hour,
            'target': val_target
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'recent': test_hour,
            'target': test_target
        },
        'X_mean': X_mean,
        'X_std': X_std
    }

    # data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
    #     'eval_target'], data_set["train_asist"], data_set["eval_asist"], data_set['train_weather'], data_set['eval_weather'], data_set['X_mean'], data_set['X_std'], \
    #     = train_input, train_target, evaluate_input, evaluate_target, train_asist_dataset, val_asist_dataset, train_weather, val_weather, X_mean, X_std

    return W_nodes, data_set

