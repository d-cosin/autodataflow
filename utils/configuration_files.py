import yaml


def parse_config_file(config_file):
    with open(config_file, "r") as stream:
        experiments_setup = yaml.safe_load(stream)
    return experiments_setup