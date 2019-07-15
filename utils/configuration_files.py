import yaml


def parse_config_file(config_file):
    with open(config_file, "r") as stream:
        parsed_file = yaml.safe_load(stream)
    return parsed_file