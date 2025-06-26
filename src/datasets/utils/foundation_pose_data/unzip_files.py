"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:24:38
Description:

"""
import os
import zipfile
from loguru import logger


def unzip_folder(path):
    files = os.listdir(path)
    for file in files:
        full_path = os.path.join(path, file)

        if os.path.isdir(full_path):
            logger.info(f"Entering directory: {full_path}")
            unzip_folder(full_path)
        elif file.endswith(".zip"):
            try:
                with zipfile.ZipFile(full_path, "r") as zip_ref:
                    logger.info(f"Unzipping {file} to {path}")
                    extract_path = os.path.splitext(full_path)[0]
                    zip_ref.extractall(extract_path)
                    logger.info(f"Done! ")
            except zipfile.BadZipFile:
                logger.error(f"Failed to unzip {full_path}: Corrupted file")
            except Exception as e:
                logger.error(f"Failed to unzip {full_path}: {e}")


path = "data/datasets/foundation_pose"

files = os.listdir(path)
logger.info(files)

unzip_folder(path)
