import json
import os

# convert config file to json
def config_to_json(config_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(config_dict, f)

# convert json to config file
def json_to_config(json_path):
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict
