import json


def load_json(json_path):
    with open(json_path, 'r') as reader:
        data = json.load(reader)
    return data

def save_json(json_path, data):
    with open(json_path, 'w') as writer:
        json.dump(data, writer)
    