import json
import lzma
import multiprocessing
import os
import random
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
import tqdm
from PIL import Image


class Dataloader:
    def __init__(self, task_zip_path: str):
        task_zip_path = os.path.expanduser(os.path.expandvars(task_zip_path))

        extract_dir = task_zip_path.replace(".zip", "")
        with zipfile.ZipFile(task_zip_path, "r") as zip_ref:
            parent_dir = os.path.dirname(extract_dir)
            zip_ref.extractall(parent_dir)

        print("Extracted to:", parent_dir)
        self.data = self.load(extract_dir)

    def _load_depth_lzma(self, depth_lzma_path):
        with open(depth_lzma_path, "rb") as f:
            compressed_data = f.read()
            decompressed = lzma.decompress(compressed_data)
            depth_array = np.frombuffer(decompressed, dtype=np.uint16).reshape(
                (480, 640)
            )
            return depth_array

    def _load_jpg(self, jpg_path):
        with open(jpg_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            return np.array(img)

    def _load_lidar_points(self, lidar_path):
        def pad_to_six(m):
            whole, dec = m.group("whole"), m.group("dec")
            return f"{whole}.{dec.ljust(6, '0')}"

        lidar_path = re.sub(
            r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_to_six, lidar_path
        )

        pcd = o3d.io.read_point_cloud(lidar_path)

        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts

    def _load_frame(self, dirpath, frame):
        depth_lzma_path = os.path.join(dirpath, frame["depth"])
        color_path = os.path.join(dirpath, frame["image"])
        lidar_path = os.path.join(dirpath, frame["lidar"])

        return {
            **frame,
            "depth": self._load_depth_lzma(depth_lzma_path),
            "image": self._load_jpg(color_path),
            "lidar": self._load_lidar_points(lidar_path),
        }

    def load(self, extract_dir):
        episodes_data = []

        for dirpath, _, filenames in tqdm.tqdm(
            os.walk(extract_dir), desc="Discovering episodes"
        ):
            if "data.json" not in filenames:
                continue

            json_path = os.path.join(dirpath, "data.json")
            with open(json_path, "r") as f:
                raw_data = json.load(f)
            print(f"Loaded JSON: {json_path}")

            max_workers = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                jobs = exe.map(lambda frame: self._load_frame(dirpath, frame), raw_data)

                episode_data = list(
                    tqdm.tqdm(
                        jobs,
                        total=len(raw_data),
                        desc=f"Loading frames for {os.path.basename(dirpath)}",
                    )
                )

            episodes_data.append(episode_data)

        return episodes_data
