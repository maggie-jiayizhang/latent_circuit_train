import json
import os

# convert config file to json
def config_to_json(config_dict, base_path, save_fn):
    with open(os.path.join(base_path, save_fn), 'w') as f:
        json.dump(config_dict, f)

# convert json to config file
def json_to_config(base_path, json_fn):
    with open(os.path.join(base_path, json_fn), 'r') as f:
        config_dict = json.load(f)
    return config_dict
