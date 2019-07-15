import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

from utils.configuration_files import parse_config_file


def split_train_test(X, y, train_size):
    n = X.shape[0]
    split_index = int(n*train_size)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return [(X_train, y_train), (X_test, y_test)]


def sliding_window(ts, target=[], lags=5):
    X = []
    y = []
    
    for i in range(len(ts)-lags):
        X.append(ts[i:i+lags])
        if len(target) > 0:
            y.append(target[i+lags])
        else:
            y.append(ts[i+lags])
    
    return np.array(X), np.array(y)


def read_dataset(path):
    df = pd.read_csv(path)
    return df


def preprocess_time_series(dataset):
    processed_time_series = dict()
    raw_dataset = read_dataset(dataset["path"])
    raw_ts = raw_dataset[dataset["features"]]
    for lag in dataset["sliding_window_lags"]:
        X, y = sliding_window(raw_ts, lags=lag)
        splitted_arrays = split_train_test(X, y, dataset["split_ratio"])
        processed_time_series["lag" + str(lag)] = splitted_arrays
    return processed_time_series


def process_datasets(datasets_setup=None):
    for dataset in datasets_setup:
        if (dataset["type"] == "time series") and (dataset["sliding_window"]):
            processed_datasets = preprocess_time_series(dataset)
    return processed_datasets


def validate_parsed_arguments(args):
    config_file = args.config_file
    if os.path.exists(config_file) == False:
        sys.exit("File does not exists. Please specify a valid path.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        help="raw config to be processed")
    args = parser.parse_args()
    validate_parsed_arguments(args)
    return args


def main():
    args = parse_arguments()
    datasets_setup = parse_config_file(args.config_file)
    processed_datasets = preprocess_datasets(datasets_setup)
    dump_datasets(processed_datasets)
    
if __name__ == "__main__":
    main()
