#!/bin/bash
###
 # @Author: Yuanhong Yu
 # @Date: 2025-03-13 20:52:54
 # @LastEditTime: 2025-03-17 21:43:33
 # @Description:
 #
###

# The script is used to make some soft links for the project.
# The script should be executed in the root directory of the project to ensure the soft links are correct.

storage_path="/input_ssd/datasets"

# Make data dir
mkdir -p data && cd data

# Create symbolic links for linemod_onepose and onepose
if [ ! -L lm ]; then
    ln -s $storage_path/linemod_onepose/lm_full ./lm
    echo "Symbolic link created: lm -> $storage_path/linemod_onepose/lm_full"
else
    echo "Symbolic link already exists: lm"
fi

if [ ! -L onepose ]; then
    ln -s $storage_path/onepose ./onepose
    echo "Symbolic link created: onepose -> $storage_path/onepose"
else
    echo "Symbolic link already exists: onepose"
fi

cd ../

output_path="/input/yyh/output/boxdreamer"

# Check and create directories for output path
if [ ! -d $output_path ]; then
    mkdir -p $output_path
    echo "Directory created: $output_path"
else
    echo "Directory already exists: $output_path"
fi

if [ ! -d $output_path/models ]; then
    mkdir -p $output_path/models
    echo "Directory created: $output_path/models"
else
    echo "Directory already exists: $output_path/models"
fi

if [ ! -d $output_path/logs ]; then
    mkdir -p $output_path/logs
    echo "Directory created: $output_path/logs"
else
    echo "Directory already exists: $output_path/logs"
fi

# Create symbolic links for models and logs
if [ ! -L models ] && [ ! -d models ]; then
    ln -s $output_path/models ./models
    echo "Symbolic link created: models -> $output_path/models"
else
    echo "Symbolic link already exists: models"
fi

if [ ! -L logs ] && [ ! -d logs ]; then
    ln -s $output_path/logs ./logs
    echo "Symbolic link created: logs -> $output_path/logs"
else
    echo "Symbolic link already exists: logs"
fi

# Make objaverse dir and create symbolic links
mkdir -p data/objaverse && cd data/objaverse

if [ ! -L bbox_3d ] && [ ! -d bbox_3d ]; then
    ln -s $output_path/objaverse_bbox3d ./bbox_3d
    echo "Symbolic link created: bbox_3d -> $output_path/objaverse_bbox3d"
else
    echo "Symbolic link already exists: bbox_3d"
fi

if [ ! -L Objaverse ]; then
    ln -s /input_ssd/datasets/Objaverse ./Objaverse
    echo "Symbolic link created: Objaverse -> /input_ssd/datasets/Objaverse"
else
    echo "Symbolic link already exists: Objaverse"
fi

if [ ! -L objaverse_render ]; then
    ln -s /input_ssd/datasets/objaverse_render ./objaverse_render
    echo "Symbolic link created: objaverse_render -> /input_ssd/datasets/objaverse_render"
else
    echo "Symbolic link already exists: objaverse_render"
fi

if [ ! -L SUN2012pascalformat ]; then
    ln -s /input_ssd/datasets/SUN2012pascalformat ./SUN2012pascalformat
    echo "Symbolic link created: SUN2012pascalformat -> /input_ssd/datasets/SUN2012pascalformat"
else
    echo "Symbolic link already exists: SUN2012pascalformat"
fi

# if /input/yyh/ok_glb_files.json exists, cp it to ./

SOURCE_FILE="/input/yyh/ok_glb_files.json"
TARGET_FILE="./ok_glb_files.json"
SOURCE_DIR="/input/yyh"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
else
    if [ ! -f "$TARGET_FILE" ]; then
        if [ -f "$SOURCE_FILE" ]; then
            cp "$SOURCE_FILE" "$TARGET_FILE"
            echo "File copied: $SOURCE_FILE to $TARGET_FILE"
        else
            echo "Source file does not exist: $SOURCE_FILE"
        fi
    else
        echo "File already exists: $TARGET_FILE"
    fi
fi


cd ../../
