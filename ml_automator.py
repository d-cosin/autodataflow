import argparse
import os.path
import sys
import yaml
import itertools

import constants 
from utils.configuration_files import parse_config_file

def mount_search_space(experiments_setup):
    search_space = list()
    for experiment in experiments_setup:
        search_definition = dict()
        search_definition["model_name"] = experiment["model_name"]
        search_definition["model_function"] = experiment["model_function"]

        hyperparameters = experiment["hyperparameters"]
        k = list(hyperparameters.keys())
        v = list(hyperparameters.values())
        hyperparameters_combinations = [t for t in itertools.product(*v)]
        for h in hyperparameters_combinations:
            d = {e[0]: e[1] for e in zip(k, h)}
            search_definition["hyperparameters"] = d
            search_space.append(search_definition.copy())
    return search_space


def execute_ml_experiment(experiments_setup, data, dataset_id):
    results = list()
    search_space = mount_search_space(experiments_setup)
    for grid_point in search_space:
        single_result = dict()
        single_result["dataset"] = dataset_id
        single_result["model"] = grid_point["model_name"]
        single_result["hyperparameters"] = grid_point["hyperparameters"]

        function = grid_point["model_function"]
        hyperparameters = grid_point["hyperparameters"]
        regressor = function(**hyperparameters).fit(data[0], data[1])
        single_result["score"] = regressor.score(data[0], data[1])

        results.append(single_result)
    return results

                
def extract_estimator_function(experiments_setup):
    for experiment in experiments_setup:
        model_name = experiment["model_name"]
        experiment["model_function"] = constants.MODEL_MAPPER[model_name]
    return experiments_setup


def validate_parsed_arguments(args):
    config_file = args.config_file
    if os.path.exists(config_file) == False:
        sys.exit("File does not exists. Please specify a valid path.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", 
                        help="configuration file with experiment setup")
    args = parser.parse_args()
    validate_parsed_arguments(args)
    return args  


def process_experiments(experiments_setup, data=None, dataset_id=None):
    experiments_setup = extract_estimator_function(experiments_setup)
    results = execute_ml_experiment(experiments_setup, data, dataset_id)
    return results

def main():
    args = parse_arguments()
    experiments_setup = parse_config_file(args.config_file)
    process_experiments(experiments_setup)

if __name__ == "__main__":
    main()