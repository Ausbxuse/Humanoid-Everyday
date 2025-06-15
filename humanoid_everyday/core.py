import json
import lzma
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d
import tqdm
from PIL import Image


class Dataloader:
    class Episode:
        def __init__(self, dirpath, frames_meta, loader):
            self.dirpath = dirpath
            self.frames_meta = frames_meta
            self.loader = loader

        def __len__(self):
            return len(self.frames_meta)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
                return self.load_frames(indices)
            if isinstance(idx, (list, tuple)):
                return self.load_frames(idx)

            if idx < 0 or idx >= len(self):
                raise IndexError("Frame index out of range")

            frame_meta = self.frames_meta[idx]
            return self.loader._load_frame(self.dirpath, frame_meta)

        def load_frames(self, indices, max_workers=None):
            def _load(i):
                if i < 0:
                    i += len(self)
                if i < 0 or i >= len(self):
                    raise IndexError(f"Frame index {i} out of range")
                frame_meta = self.frames_meta[i]
                return self.loader._load_frame(self.dirpath, frame_meta)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # map preserves order of `indices`
                return list(executor.map(_load, indices))

        def __iter__(self):
            for idx in range(len(self)):
                yield self[idx]

    def __init__(self, task_zip_path: str):
        task_zip_path = os.path.expanduser(os.path.expandvars(task_zip_path))
        extract_dir = task_zip_path.replace(".zip", "")

        parent_dir = os.path.dirname(extract_dir)
        print("Extracting to", extract_dir)
        with zipfile.ZipFile(task_zip_path, "r") as zip_ref:
            members = [
                m for m in zip_ref.namelist() if m and not m.startswith("__MACOSX")
            ]
            root_dirs = set([m.split("/")[0] for m in members])

            zip_ref.extractall(parent_dir)

        basename = os.path.basename(extract_dir)
        if not (os.path.isdir(extract_dir) and root_dirs == {basename}):
            print(
                f"No root folder '{basename}' found. Creating task directory '{extract_dir}'."
            )
            os.makedirs(extract_dir, exist_ok=True)
            for root in root_dirs:
                src = os.path.join(parent_dir, root)
                dst = os.path.join(extract_dir, root)
                if os.path.exists(src) and src != extract_dir:
                    os.rename(src, dst)
                    # print(f"Moved '{src}' to '{dst}'")

        self.episodes_index = []
        for dirpath, _, filenames in os.walk(extract_dir):
            if "data.json" not in filenames:
                continue

            json_path = os.path.join(dirpath, "data.json")
            with open(json_path, "r") as f:
                raw_data = json.load(f)
            # print(f"Indexed JSON: {json_path}")

            self.episodes_index.append((dirpath, raw_data))
        print(f"Finished indexing {len(self.episodes_index)} episodes.")

    def __len__(self):
        return len(self.episodes_index)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Episode index out of range")
        dirpath, frames_meta = self.episodes_index[idx]
        return Dataloader.Episode(dirpath, frames_meta, self)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

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
        depth_path = os.path.join(dirpath, frame["depth"])
        image_path = os.path.join(dirpath, frame["image"])
        lidar_path = os.path.join(dirpath, frame["lidar"])

        return {
            **frame,
            "depth": self._load_depth_lzma(depth_path),
            "image": self._load_jpg(image_path),
            "lidar": self._load_lidar_points(lidar_path),
        }

    # TODO: add H1/G1 cam intrinsics
    def display_depth_point_cloud(self, eps_idx, step_idx):
        depth_array = self[eps_idx][step_idx]["depth"]
        fx = 389.07278
        fy = 389.07278
        cx = 321.61887
        cy = 238.43630
        h, w = depth_array.shape
        pts = []
        for v in range(h):
            for u in range(w):
                Z = depth_array[v, u]
                if Z > 0:
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    pts.append([X, Y, Z])
        pts = np.array(pts)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)
        o3d.visualization.draw_geometries([pc, coord])

    def display_lidar_point_cloud(self, eps_idx, step_idx):
        lidar_points = self[eps_idx][step_idx]["lidar"]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        pc.paint_uniform_color([0.5, 0.5, 0.5])
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)
        o3d.visualization.draw_geometries([pc, coord])

    def display_image(self, eps_idx, step_idx):
        image_array = self[eps_idx][step_idx]["image"]
        img = Image.fromarray(image_array)
        img.show()
