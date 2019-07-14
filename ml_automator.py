import argparse
import os.path
import sys
import yaml
import itertools

import constants 


from sklearn.datasets import load_boston
boston = load_boston()

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


def execute_ml_experiment(experiments_setup):
    search_space = mount_search_space(experiments_setup)
    for grid_point in search_space:
        function = grid_point["model_function"]
        hyperparameters = grid_point["hyperparameters"]
        regressor = function(**hyperparameters).fit(boston.data, boston.target)
        print(regressor)

                
def extract_estimator_function(experiments_setup):
    for experiment in experiments_setup:
        model_name = experiment["model_name"]
        experiment["model_function"] = constants.MODEL_MAPPER[model_name]
    return experiments_setup


def parse_config_file(config_file):
    with open(config_file, "r") as stream:
        experiments_setup = yaml.safe_load(stream)
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


def main():
    args = parse_arguments()
    experiments_setup = parse_config_file(args.config_file)
    experiments_setup = extract_estimator_function(experiments_setup)
    execute_ml_experiment(experiments_setup)

if __name__ == "__main__":
    main()