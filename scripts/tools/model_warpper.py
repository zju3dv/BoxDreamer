'read the model checkpoints and split the model into diferenet parts, support interation with user to select the parts of the model to be used'

import torch 
from src.utils.log import *
import os
import argparse

def get_model_parts(model_path):
    model = torch.load(model_path)
    model_parts = {}
    if isinstance(model, dict):
        for key in model.keys():
            model_parts[key] = model[key]
    else:
        INFO('Model is not a dictionary, it is a ' + str(type(model)))
        INFO(f"Parts of the model are: {model}")
        
    return model_parts

def dump_model_parts(model_parts, root_path):
    for key in model_parts.keys():
        filename = os.path.join(root_path, key + '.pth')
        torch.save(model_parts[key], filename)

def model_keys(model_parts):
    keys = set()
    for key in model_parts.keys():
        keys.add(key.split('.')[0])
        # INFO(key)
        if key.split('.')[0] in model_parts.keys() and isinstance(model_parts[key.split('.')[0]], dict):
            for sub_key in model_parts[key.split('.')[0]].keys():
                keys.add(key.split('.')[0] + '.' + sub_key)
                
        
    return keys

def main(path):
    model_parts = get_model_parts(path)
    # show the parts of the model, and ask the user to select the parts to be used
    INFO('Model parts are loaded')
    INFO('Model parts are:')
    keys = model_keys(model_parts)
    for key in keys:
        INFO(key)
    selected_parts = {}
    while True:
        part = input('Enter the part to be used (Enter "done" to finish): ')
        if part == 'done':
            break
        selected_parts[part] = []
        for key in model_parts.keys():
            if key.startswith(part):
                selected_parts[part].append(key)
            

        
    root_path = os.path.dirname(path)
    dump_model_parts(selected_parts, root_path)
    
    INFO('Selected parts are saved in the same directory as the model')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, help='Path to the model')
    args = args.parse_args()
    
    main(args.path)
    
    