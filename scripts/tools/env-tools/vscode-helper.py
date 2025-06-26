# sometimes some well-known plugins are not installed or not enabled successfully in vscode, this script is used to install and enable them.
import os
import sys
import argparse

def pylance_helper() -> None:
    '''
        pylance will crash if the workspace is too large, this function will help to make pylance work again.
    '''
    
    # ask user to get root workspace path
    # default is the current workspace path
    root_path = os.getcwd()
    print(f"Current workspace path: {root_path}")
    root_path = input("Please input the root workspace path (default is the current path), press enter to use default: ")
    if not root_path:
        root_path = os.getcwd()
        
    # get all files in workspace, sugget some files to include
    useful_formats = ['.py', '.ipynb', 'yaml', 'json']
    exclude_folders = ['__pycache__', '.git', '.vscode', '.ipynb_checkpoints']
    
    # count most useful dirs to make pylance work
    dir_count = {}
    for root, _, files in os.walk(root_path):
        if any([folder in root for folder in exclude_folders]):
            continue
        for file in files:
            if file.endswith(tuple(useful_formats)):
                dir_count[root] = dir_count.get(root, 0) + 1
                
    # parse the dir_count, accumulate the count of subdirs , only subdir of root_path is counted
    def get_parent_dir(dir: str) -> str:
        return os.path.dirname(dir)
    
    for dir in list(dir_count.keys()):
        if dir == root_path:
            continue
        parent_dir = get_parent_dir(dir)
        while parent_dir != root_path:
            dir_count[parent_dir] = dir_count.get(parent_dir, 0) + dir_count[dir]
            parent_dir = get_parent_dir(parent_dir)
    
    # delete non-root_path's subdirs' count
    for dir in list(dir_count.keys()):
        if get_parent_dir(dir) != root_path:
            del dir_count[dir]
    
    # sort by count
    dir_count = dict(sorted(dir_count.items(), key=lambda x: x[1], reverse=True))
    
    # log most useful dirs for users to select
    print("Most useful dirs:")
    for idx, (dir, count) in enumerate(dir_count.items()):
        print(f"{idx}: {dir} ({count} files)")
        
    # ask user to select dirs
    selected_dirs = input("Please input the dirs you want to include, split by comma: ")
    selected_dirs = selected_dirs.split(',')
    selected_dirs = [list(dir_count.keys())[int(idx)] for idx in selected_dirs]
    # del selected dirs prefix root_path
    selected_dirs = [os.path.relpath(dir, root_path) for dir in selected_dirs]
    
    # generate root/pyrightconfig.json
    pyright_config = {
        "include": selected_dirs,
    }
    
    # write to root/pyrightconfig.json
    with open(os.path.join(root_path, 'pyrightconfig.json'), 'w') as f:
        import json
        f.write(json.dumps(pyright_config, indent=4))
        
    print(f"pyrightconfig.json is generated at {root_path}.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vscode helper.')
    parser.add_argument('plugin', type=str, help='plugin name (you want to get help with)')
    args = parser.parse_args()
    
    # call corresponding function
    function_name = f"{args.plugin}_helper"
    if function_name in globals():
        globals()[function_name]()
    else:
        print(f"Plugin {args.plugin} is not supported.")
    
        