"""
Author: Yuanhong Yu
Date: 2025-03-17 15:23:09
LastEditTime: 2025-03-17 15:23:26
Description: Google Drive Downloader for FoundationPose Dataset

"""
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import io
from googleapiclient.http import MediaIoBaseDownload

# https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-

SCOPES = ["https://www.googleapis.com/auth/drive"]
os.environ["HTTP_PROXY"] = "http://127.0.0.1:15777"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:15777"

flow = InstalledAppFlow.from_client_secrets_file(
    "/mnt/workspace/yuyuanhong/code/OnePoseV3/src/foundation_pose_data/credentials.json",
    SCOPES,
)
creds = flow.run_local_server(port=0, open_browser=True)
service = build("drive", "v3", credentials=creds)


def list_files_in_folder(folder_id):
    try:
        query = f"'{folder_id}' in parents"
        results = (
            service.files()
            .list(q=query, fields="files(id, name, mimeType, size)")
            .execute()
        )
        return results.get("files", [])
    except Exception as e:
        print(f"Failed to list files: {e}")
        return []


def download_file(file_id, file_path, file_size):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == file_size:
            print(f"File already downloaded: {file_path}")
            return
        else:
            os.remove(file_path)

    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()
        print(f"Downloaded {file_path}")
    except Exception as e:
        print(f"Failed to download file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)


def download_folder(folder_id, local_path):
    os.makedirs(local_path, exist_ok=True)
    files = list_files_in_folder(folder_id)
    for file in files:
        file_path = os.path.join(local_path, file["name"])
        if file["mimeType"] == "application/vnd.google-apps.folder":
            download_folder(file["id"], file_path)
        else:
            download_file(file["id"], file_path, int(file.get("size", 0)))


top_folder_id = "1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-"
local_directory = (
    "/home/admin/workspace/yuyuanhong/data/oneposev3/datasets/foundation_pose"
)
download_folder(top_folder_id, local_directory)
