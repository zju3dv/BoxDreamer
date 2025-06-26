"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:19:12
Description: Objaverse dataset downloader

"""
import objaverse.xl as oxl
import objaverse
from objaverse.utils import get_uid_from_str
from loguru import logger
import http.client
from time import sleep
import os
import sys
import json
import random
import platform
import subprocess
import concurrent.futures
import glob
import torch
from rich.progress import Progress
import shutil
import contextlib
import io
from tqdm import tqdm

# check the working directory
print(os.getcwd())
sys.path.append("src")
from utils.log import *


class ObjaverseHandler:
    def __init__(self, config):
        self.download_dir = config["download_dir"]
        self.render_dir = config["render_dir"]
        self.version = config["version"]  # "v1 or xl"
        proxy = config.get("proxy", None)
        use_proxy = config.get("use_proxy", False)
        if use_proxy:
            INFO(f"Proxy is enabled. Setting proxy to {proxy}...")
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy

            if not self.check_proxy():
                raise Exception(
                    "Proxy is not working. Please check the proxy settings."
                )

        self.download_cfg = config["download_cfg"]
        self.shuffle = config.get("shuffle", False)
        self.download_obj_ids = set()
        self.local_file_mapping = {}

        self.predownload_path = config.get("predownload_path", None)
        self.predownload = None

        self.render_cfg = config.get("render", None)

    def check_proxy(self):
        # check if the proxy is working
        test_url = "www.google.com"
        try:
            http.client.HTTPSConnection(test_url)
            INFO("Proxy is working.")
            return True
        except Exception as e:
            ERROR(e)
            return False

    def filter_annotations(self):
        annotations = self.annotations
        source = self.download_cfg.get("source", None)  # list of sources to download
        INFO(f"Filtering annotations by source: {source}")
        if source:
            annotations = annotations[annotations["source"].isin(source)]

        self.annotations = annotations

    def get_obj_alignment_annotations(self):
        annotations = oxl.get_alignment_annotations(download_dir=self.download_dir)
        self.annotations = annotations

        self.filter_annotations()

        return self.annotations

    def get_obj_annotations(self):
        annotations = oxl.get_annotations(download_dir=self.download_dir)
        self.annotations = annotations

        self.filter_annotations()

        return self.annotations

    def get_lvis_objs(self):
        lvis_objs = objaverse.load_lvis_annotations()  # dict type

        self.lvis_objs = []
        for k, v in lvis_objs.items():
            self.lvis_objs.extend(v)

        INFO(f"Total objects with lvis category: {len(self.lvis_objs)}")

        return self.lvis_objs

    def get_lvis_annotations(self):
        # search lvis objs uid from annotations

        lvis_objs = self.get_lvis_objs()
        if hasattr(self, "annotations"):
            annotations = self.annotations
        else:
            annotations = self.get_obj_annotations()

        assert (
            annotations["source"] == "sketchfab"
        ).all(), "Only sketchfab objects are supported."

        # attention : uid is not the same as fileIdentifier, uid is the last part of fileIdentifier(obj file prefix)
        file_names = (
            annotations["fileIdentifier"].str.split("/").str[-1].str.split(".").str[0]
        )
        lvis_annotations = annotations[file_names.isin(lvis_objs)]

        INFO(f"Found {len(lvis_annotations)} objects with lvis category.")

        self.annotations = lvis_annotations

        return self.annotations

    def get_annotations_metadata(self):
        annotations = self.annotations
        INFO(f"Found {len(annotations)} objects.")
        INFO(f"Columns: {annotations.columns}")
        metadata = annotations["metadata"]

        # log first 5 metadata
        for i in range(5):
            INFO(f"Metadata {i}: {metadata.iloc[i]}")

    def escape_bash_string(self, s: str) -> str:
        """Escapes special characters in a string for use in bash command
        lines.

        Args:
        s (str): The original string to escape.

        Returns:
        str: The escaped string suitable for bash commands.
        """
        # Characters to escape
        # You might need to add more characters here depending on the context
        bash_special_chars = [
            "(",
            ")",
            "&",
            "|",
            ";",
            "<",
            ">",
            "`",
            "$",
            '"',
            "'",
            "{",
            "}",
        ]

        # Escaping each special character with a backslash
        escaped_string = s
        for char in bash_special_chars:
            escaped_string = escaped_string.replace(char, f"\\{char}")

        return escaped_string

    def init_local_objs(self):
        self.load_predownload()
        self.find_predownload()
        self.check_predownload()

        return self.predownload

    def load_predownload(self):
        # load json file (mapping between uuid and local path)
        if os.path.exists(self.predownload_path):
            self.predownload = json.load(open(self.predownload_path, "r"))
            INFO(f"Loaded {len(self.predownload)} pre-downloaded objects.")
        else:
            INFO("No pre-download mapping found.")

    def _get_all_files(self, directory):
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                all_files.append(relative_path)
        return all_files

    def _get_uid(self, source, fileIdentifier):
        if source == "thingiverse":
            uid = f"thing-{fileIdentifier.split('/')[-2].split(':')[-1]}-file-{fileIdentifier.split('=')[-1]}"
        elif source == "sketchfab":
            uid = fileIdentifier.split("/")[-1]
        else:
            uid = get_uid_from_str(fileIdentifier)

        return uid

    def prepare_uid_map(self, annotations):
        uid_to_uuid = {}
        uuid_to_iden = {}

        def get_uids(row):
            uid = self._get_uid(row["source"], row["fileIdentifier"])
            uuid = get_uid_from_str(row["fileIdentifier"])
            return uid, uuid, row["fileIdentifier"]

        tqdm.pandas(desc="Prepare uid map")
        uid_uuid_pairs = annotations.progress_apply(get_uids, axis=1)

        with Progress() as progress:
            task = progress.add_task("Prepare uid map", total=len(uid_uuid_pairs))
            for idx, (uid, uuid, iden) in enumerate(uid_uuid_pairs):
                uid_to_uuid[uid] = uuid
                uuid_to_iden[uuid] = iden
                progress.update(task, advance=1)

        return uid_to_uuid, uuid_to_iden

    def find_predownload(self):
        root_dir = self.download_dir
        sources = self.download_cfg.get("source", [])
        annotations = self.get_obj_annotations()
        objs_mapping = {}

        source_dirs = {
            "sketchfab": "hf-objaverse-v1/glbs",
            "smithsonian": "smithsonian/objects",
            "thingiverse": "thingiverse",
        }
        INFO(f"Searching pre-downloaded objects in {root_dir}...")

        INFO(f"Established annotation mapping")
        uid_to_uuid, uuid_to_iden = self.prepare_uid_map(annotations)

        def process_files(source, obj_dir):
            local_objs_mapping = {}
            INFO(f"Processing files in {obj_dir}...")

            files = self._get_all_files(obj_dir)
            with Progress() as progress:
                task = progress.add_task(
                    f"Processing files in {obj_dir}...", total=len(files)
                )
                for f in files:
                    if source == "thingiverse" and f.endswith(".parquet"):
                        continue
                    uid = f.split("/")[-1].split(".")[0]
                    # INFO(f"Found object {uid} from source {source}.")
                    if uid in uid_to_uuid.keys():
                        uuid = uid_to_uuid[uid]
                        local_objs_mapping[uuid] = {
                            "local_path": os.path.join(obj_dir, f),
                            "source": source,
                            "fileIdentifier": uuid_to_iden[uuid],
                        }
                    progress.update(task, advance=1)

            return local_objs_mapping

        for source, subdir in source_dirs.items():
            if source in sources:
                obj_dir = os.path.join(root_dir, subdir)
                if os.path.exists(obj_dir):
                    local_objs_mapping = process_files(source, obj_dir)
                    objs_mapping.update(local_objs_mapping)
                else:
                    WARNING(f"Directory {obj_dir} does not exist.")

        if "github" in sources:
            ERROR(
                "Github source is not supported for searching pre-downloaded objects."
            )

        INFO(f"Found {len(objs_mapping)} pre-downloaded objects.")

        if hasattr(self, "predownload"):
            if self.predownload is not None:
                missing_objs = set(objs_mapping.keys()) - set(self.predownload.keys())
            else:
                missing_objs = objs_mapping

            INFO(f"Found {len(missing_objs)} missing objects.")
            if self.predownload is not None:
                self.predownload.update(objs_mapping)
            else:
                self.predownload = objs_mapping

    def check_predownload(self):
        # check if the pre-download mapping is correct and available for access
        if self.predownload is None:
            WARNING("No pre-download mapping found.")
            return False
        else:
            checked_mapping = {}
            with Progress() as progress:
                task = progress.add_task(
                    "Checking pre-downloaded objects...", total=len(self.predownload)
                )
                for k, v in self.predownload.items():
                    if not os.path.exists(v["local_path"]):
                        WARNING(f"Pre-downloaded object {v} not found.")
                        exit(1)
                        continue
                    else:
                        progress.update(task, advance=1)
                        checked_mapping[k] = v

        self.predownload = checked_mapping

        # dump the checked mapping
        if os.path.exists(self.predownload_path):
            os.remove(self.predownload_path)

        with open(self.predownload_path, "w") as f:
            json.dump(checked_mapping, f)

        INFO(f"{len(checked_mapping)} pre-downloaded objects are available.")

    def select_gpu(self):
        # return the gpu id based on the number of gpus available and memory consumption
        available_gpus = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        INFO(f"Available GPUs: {available_gpus}")
        INFO(f"GPU memory: {gpu_memory}")
        consumed_memory = dict()
        for id in range(available_gpus):
            consumed_memory[id] = torch.cuda.memory_allocated(id)

        sorted_consumed_memory = sorted(consumed_memory.items(), key=lambda x: x[1])

        return sorted_consumed_memory[0][0]

    def key_interrupt_handler(self):
        INFO("Keyboard interrupt detected. Exiting...")
        # in case of keyboard interrupt, save the downloaded files and then exit
        self.dump_download_mapping()
        sys.exit()

    def download(self):
        """
        update:  local file mapping (maybe for rendering)
        """
        if self.shuffle:
            self.annotations = self.annotations.sample(frac=1)
        # signal.signal(signal.SIGINT, self.key_interrupt_handler)

        data_range = self.download_cfg.get("range", None)
        obj = self.annotations
        if data_range and data_range != "all":
            start, end = data_range
            obj = obj[start:end]

        INFO(f"Download {len(obj)} objects to {self.download_dir}...")
        if len(obj) == 0:
            INFO("No objects to download.")
            return

        object_ids = set(obj["fileIdentifier"])
        obj_uuids = set([get_uid_from_str(i) for i in object_ids])

        processes = self.download_cfg.get("processes", 1)
        while True:
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    ret = oxl.download_objects(
                        objects=obj,
                        download_dir=self.download_dir,
                        processes=processes,
                        save_repo_format="files",
                    )
                del f
                # ret -> uuid : local path
                if ret.__len__() == 0:
                    INFO("No objects downloaded.")
                else:
                    # INFO(f"ret: {ret}")

                    local_file_mapping = {}
                    for k, v in ret.items():
                        uid = get_uid_from_str(k)
                        local_file_mapping[uid] = {}
                        local_file_mapping[uid]["local_path"] = v
                        local_file_mapping[uid]["source"] = obj[
                            obj["fileIdentifier"] == k
                        ]["source"].values[0]
                        local_file_mapping[uid]["fileIdentifier"] = k

                    self.local_file_mapping.update(local_file_mapping)
                    ret = {get_uid_from_str(k): v for k, v in ret.items()}
                    obj_ids = set(ret.keys())
                    self.download_obj_ids.update(obj_ids)
                    break
            except KeyboardInterrupt:
                self.key_interrupt_handler()

            except Exception as e:
                ERROR(e)
                INFO("Retrying download...")
                sleep(30)
                # avoid recursion and stack overflow
                # self.download()
                continue

        INFO(f"Downloaded {len(self.download_obj_ids)} objects.")

        if self.predownload is not None:
            # update local path by querying the pre-download mapping
            not_downloaded_uuid = obj_uuids - self.download_obj_ids
            already_downloaded_mapping = {
                k: self.predownload[k]
                for k in not_downloaded_uuid
                if k in self.predownload
            }
            INFO(f"Found {len(already_downloaded_mapping)} objects already downloaded.")
            self.local_file_mapping.update(already_downloaded_mapping)

        self.dump_download_mapping()

        return self.local_file_mapping

    def dump_download_mapping(self):
        # delete the download mapping file if it exists
        if os.path.exists(os.path.join(self.download_dir, "download_mapping.json")):
            already_downloaded = json.load(
                open(os.path.join(self.download_dir, "download_mapping.json"), "r")
            )
        else:
            already_downloaded = {}
        with open(os.path.join(self.download_dir, "download_mapping.json"), "w") as f:
            # local file mapping is a dictionary of uuid: local path
            already_downloaded.update(self.local_file_mapping)
            # uuid : local path
            json.dump(already_downloaded, f, indent=2)

        INFO(f"Download mapping saved to {self.download_dir}.")

    def render_single_object(self, id, info, render_cfg, render_dir):
        # id: uuid
        save_uid = get_uid_from_str(info["fileIdentifier"])
        local_path = info["local_path"]
        source = info["source"]
        # INFO(f"Rendering object {save_uid}...")
        local_path = self.escape_bash_string(local_path)
        args = f"--object_path '{local_path}' --num_renders {render_cfg['num_renders']}"

        gpu_devices = render_cfg.get("gpu_devices", 0)
        using_gpu = render_cfg.get("using_gpu", False)
        if isinstance(gpu_devices, int) and gpu_devices > 0:
            num_gpus = gpu_devices
            gpu_i = random.randint(0, num_gpus - 1)
        elif isinstance(gpu_devices, list):
            gpu_i = random.choice(gpu_devices)
        elif isinstance(gpu_devices, int) and gpu_devices == 0:
            using_gpu = False

        output_dir = os.path.join(render_dir, source, save_uid)
        os.makedirs(output_dir, exist_ok=True)
        args += f" --output_dir {output_dir}"
        # todo fix this
        if platform.system() == "Linux" and using_gpu:
            args += " --engine CYCLES"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            args += " --engine CYCLES"

        only_northern_hemisphere = render_cfg.get("only_northern_hemisphere", False)
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        blender_path = render_cfg.get("blender_path", "blender")
        script_path = render_cfg.get(
            "script_path", "src/objaverse_utils/scripts/rendering.py"
        )
        render_cmd = f"\
            {blender_path} -b --python {script_path} -- {args}"

        if using_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_i)

        result = subprocess.run(
            ["bash", "-c", render_cmd],
            timeout=render_cfg.get("render_timeout", 60),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output = result.stdout.decode("utf-8")
        errors = result.stderr.decode("utf-8")

        logger.info(f"Output: {output}")
        if errors.strip() != "":
            logger.info(f"Errors: {errors}")

        # update the metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            ERROR(f"Metadata file not found for object {id}!")
            return (id, output_dir, result.returncode)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        metadata_file["file_identifier"] = info["fileIdentifier"]
        metadata_file["save_uid"] = save_uid
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(output_dir, "rgb", "*.png"))
        metadata_files = glob.glob(os.path.join(output_dir, "*.json"))
        npy_files = glob.glob(os.path.join(output_dir, "camera_pose", "*.npy"))

        if (
            (len(png_files) != self.render_cfg["num_renders"])
            or (len(npy_files) != self.render_cfg["num_renders"])
            or (len(metadata_files) != 1)
        ):
            logger.error(f"Found object {id} was not rendered successfully!")

            return (id, output_dir, result.returncode)

        return (save_uid, output_dir, result.returncode)

    def set_render_objs(self, render_objs=None):
        self.render_objs = render_objs
        if render_objs is None:
            self.render_objs = self.predownload
        elif self.predownload is None:
            self.render_objs = self.local_file_mapping

    def render_parallel(self):
        INFO(f"Rendering {len(self.render_objs)} objects...")
        self.renderd_mapping = {}
        with Progress() as progress:
            task = progress.add_task(
                "Rendering objects...", total=len(self.render_objs)
            )

            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        self.render_single_object,
                        id,
                        info,
                        self.render_cfg,
                        self.render_dir,
                    )
                    for id, info in self.render_objs.items()
                ]

                for future in concurrent.futures.as_completed(futures):
                    id, output_dir, return_code = future.result()
                    if return_code == 0:
                        # INFO(f"Rendering of object {id} completed successfully.")
                        self.renderd_mapping[id] = output_dir
                        progress.update(task, advance=1)
                    else:
                        ERROR(f"Failed to render object {id}.")
                        shutil.rmtree(output_dir)
                        progress.update(task, advance=1)

        # save the render mapping
        INFO(f"Saving render mapping to {self.render_dir}...")
        INFO(f"{len(self.renderd_mapping)} objects were rendered.")
        if os.path.exists(os.path.join(self.render_dir, "render_mapping.json")):
            already_rendered = json.load(
                open(os.path.join(self.render_dir, "render_mapping.json"), "r")
            )
            self.renderd_mapping.update(already_rendered)

        with open(os.path.join(self.render_dir, "render_mapping.json"), "w") as f:
            json.dump(self.renderd_mapping, f, indent=2, sort_keys=True)
