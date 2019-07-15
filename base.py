import argparse
import os
import sys

from utils.configuration_files import parse_config_file
from data_automator import process_datasets
from ml_automator import process_experiments


def validate_parsed_arguments(args):
    ml_config_file = args.ml_config_file
    data_config_file = args.data_config_file
    files = [ml_config_file, data_config_file]
    for file in files:
        if os.path.exists(file) == False:
            sys.exit("File does not exists. Please specify a valid path.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_config_file",
                        help="configuration file with ml experiment setup")
    parser.add_argument("--data_config_file",
                        help="configuration file with data processing")
    args = parser.parse_args()
    validate_parsed_arguments(args)
    return args


def main():
    args = parse_arguments()
    datasets_setup = parse_config_file(args.data_config_file)
    processed_datasets = process_datasets(datasets_setup)
    experiments_setup = parse_config_file(args.ml_config_file)
    process_results = list()
    for dataset_id, dataset_arrays in processed_datasets.items():
        train_data, test_data = dataset_arrays
        results = process_experiments(experiments_setup, train_data, dataset_id) # TODO: criar objeto dataset e passar o nome por ele
        process_results.extend(results)
    print(process_results)

if __name__ == "__main__":
    main()