import argparse
import json
import lzma
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import open3d as o3d
import pandas as pd
from datasets import Array2D, Dataset, Features, Sequence, Value
from huggingface_hub import create_repo, create_tag, upload_large_folder
from tqdm import tqdm

CODE_VERSION = "2.0.1"
FPS = 30


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


def read_json_list(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except Exception:
        return None


def iter_tasks(data_root: Path) -> Iterator[Tuple[str, Path, str, str]]:
    for cat_dir in sorted(
        [p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    ):
        for task_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            yield f"{cat_dir.name}/{task_dir.name}", task_dir, cat_dir.name, task_dir.name


class HE2LeRobotConverter:
    def __init__(self):
        self.features = Features(
            {
                "observation.depth.egocentric": Array2D(dtype="float32", shape=(480, 640)),
                "observation.lidar": Sequence(Sequence(Value("float32"))),
                "observation.imu.quaternion": Sequence(Value("float32")),
                "observation.imu.accelerometer": Sequence(Value("float32")),
                "observation.imu.gyroscope": Sequence(Value("float32")),
                "observation.imu.rpy": Sequence(Value("float32")),
                "observation.odometry.position": Sequence(Value("float32")),
                "observation.odometry.velocity": Sequence(Value("float32")),
                "observation.odometry.rpy": Sequence(Value("float32")),
                "observation.odometry.quat": Sequence(Value("float32")),
                "observation.arm_joints": Sequence(Value("float32")),
                "observation.leg_joints": Sequence(Value("float32")),
                "observation.hand_joints": Sequence(Value("float32")),
                "observation.tactile": Array2D(dtype="float32", shape=(-1, 4)),
                "action": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )

        self.task_description_dict: Dict[str, str] = {}
        self.kept_records: List[Tuple[int, int, Path, str, str]] = []
        self.lengths_by_episode: Dict[int, int] = {}
        self.tasks_meta: Dict[int, Dict[str, Any]] = {}
        self.global_stats: Dict[str, Optional[Dict[str, Any]]] = {"h1": None, "g1": None}
        self.num_episodes: int = 0
        self.total_frames: int = 0
        self.chunks_size: int = 1000

    def get_robot_type(self, ep_dir: Path) -> str:
        data_list = read_json_list(ep_dir / "data.json")
        if not data_list:
            return "h1"
        for frame in data_list:
            st = frame.get("states", {})
            if isinstance(st, dict) and "robot_type" in st:
                try:
                    return str(st["robot_type"]).lower()
                except Exception:
                    return "h1"
            if "robot_type" in frame:
                try:
                    return str(frame["robot_type"]).lower()
                except Exception:
                    return "h1"
        return "h1"

    def load_depth(self, depth_lzma_path: Path) -> Optional[np.ndarray]:
        try:
            with open(depth_lzma_path, "rb") as f:
                decompressed = lzma.decompress(f.read())
            depth_u16 = np.frombuffer(decompressed, dtype=np.uint16).reshape((480, 640))
            return depth_u16.astype(np.float32)
        except Exception:
            return None

    def load_lidar(self, pcd_path: Path) -> Optional[np.ndarray]:
        try:

            def pad_to_six(m):
                whole, dec = m.group("whole"), m.group("dec")
                return f"{whole}.{dec.ljust(6, '0')}"

            pcd_path_str = re.sub(r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_to_six, str(pcd_path))
            pcd = o3d.io.read_point_cloud(pcd_path_str)
            pts = np.asarray(pcd.points, dtype=np.float32)
            pts = pts[~np.all(pts == 0, axis=1)]
            if pts.size == 0:
                return None
            return pts
        except Exception:
            return None

    def load_tactile(self, state: Dict[str, Any]) -> List[Any]:
        sensor_ids, values = [], []
        if state and isinstance(state, dict) and "hand_pressure_state" in state and state["hand_pressure_state"] is not None:
            for sensor in state["hand_pressure_state"]:
                sid = int(sensor.get("sensor_id", -1))
                raw_vals = [float(x) for x in sensor.get("usable_readings", [])]
                if len(raw_vals) < 4:
                    raw_vals += [math.nan] * (4 - len(raw_vals))
                elif len(raw_vals) > 4:
                    raw_vals = raw_vals[:4]
                sensor_ids.append(sid)
                values.append(raw_vals)
        return values

    def build_obs(self, frame: Dict[str, Any], depth_arr: np.ndarray, pts: np.ndarray) -> Dict[str, Any]:
        states = frame.get("states", {}) or {}

        imu_in = states.get("imu", {}) if isinstance(states, dict) else {}
        imu = {"quaternion": [np.nan] * 4, "accelerometer": [np.nan] * 3, "gyroscope": [np.nan] * 3, "rpy": [np.nan] * 3}
        for k, n in (("quaternion", 4), ("accelerometer", 3), ("gyroscope", 3), ("rpy", 3)):
            if k in imu_in and isinstance(imu_in[k], (list, tuple)):
                vals = [float(x) for x in imu_in[k][:n]]
                imu[k] = vals + [0.0] * (n - len(vals))

        odo_in = states.get("odometry", {}) if isinstance(states, dict) else {}
        odometry = {"position": [np.nan] * 3, "velocity": [np.nan] * 3, "rpy": [np.nan] * 3, "quat": [np.nan] * 4}
        for k, n in (("position", 3), ("velocity", 3), ("rpy", 3), ("quat", 4)):
            if k in odo_in and isinstance(odo_in[k], (list, tuple)):
                vals = [float(x) for x in odo_in[k][:n]]
                odometry[k] = vals + [0.0] * (n - len(vals))

        arm_joints = [float(x) for x in states.get("arm_state", [])]
        leg_joints = [float(x) for x in states.get("leg_state", [])]
        hand_joints = [float(x) for x in states.get("hand_state", [])]

        tactile = self.load_tactile(states)

        return {
            "observation.depth.egocentric": depth_arr.tolist(),
            "observation.lidar": pts.tolist(),
            "observation.imu.quaternion": imu["quaternion"],
            "observation.imu.accelerometer": imu["accelerometer"],
            "observation.imu.gyroscope": imu["gyroscope"],
            "observation.imu.rpy": imu["rpy"],
            "observation.odometry.position": odometry["position"],
            "observation.odometry.velocity": odometry["velocity"],
            "observation.odometry.rpy": odometry["rpy"],
            "observation.odometry.quat": odometry["quat"],
            "observation.arm_joints": arm_joints,
            "observation.leg_joints": leg_joints,
            "observation.hand_joints": hand_joints,
            "observation.tactile": tactile,
        }

    def build_act(self, frame: Dict[str, Any]) -> List[float]:
        hand_joints: List[float] = []
        arm_joints: List[float] = []
        actions = frame.get("actions", {}) or {}
        r = actions.get("right_angles")
        l = actions.get("left_angles")

        def convert_h1_hand(qpos: List[float]) -> List[float]:
            out = [1.7 - qpos[i] for i in [4, 6, 2, 0]]
            out.append(1.2 - qpos[8])
            out.append(0.5 - qpos[9])
            return [float(x) for x in out]

        if r is not None and l is not None:
            if len(r) == 12 and len(l) == 12:
                hand_joints.extend(convert_h1_hand(l))
                hand_joints.extend(convert_h1_hand(r))
            elif len(r) == 7 and len(l) == 7:
                hand_joints.extend([float(x) for x in l])
                hand_joints.extend([float(x) for x in r])

        sq = actions.get("sol_q")
        if sq is not None:
            arm_joints = [float(x) for x in sq]
        return hand_joints + arm_joints

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        episode_dir: Path,
        out_base: Path,
        chunks_size: int,
    ) -> Tuple[int, int, Dict[str, Any]]:
        chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

        data_list = read_json_list(episode_dir / "data.json")
        assert data_list is not None and isinstance(data_list, list), f"data.json malformed in {episode_dir}"

        rows: List[Dict[str, Any]] = []
        rgb_paths: List[Path] = []
        for i, frame in enumerate(data_list):
            rgb = frame["image"]
            rgb_paths.append((episode_dir / rgb).resolve())

            depth_arr = self.load_depth((episode_dir / frame["depth"]).resolve())
            if depth_arr is None:
                depth_arr = np.full((480, 640), np.nan, dtype=np.float32)

            lidar_pts = self.load_lidar((episode_dir / frame["lidar"]).resolve())
            if lidar_pts is None:
                lidar_pts = np.zeros((0, 3), dtype=np.float32)

            obs = self.build_obs(frame, depth_arr, lidar_pts)
            act = self.build_act(frame)

            rows.append(
                {
                    **obs,
                    "action": act,
                    "timestamp": i * (1.0 / FPS),
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": i,  # TODO: global index if needed
                    "task_index": task_index,
                    "next.done": (i == len(data_list) - 1),
                }
            )

        assert rows, f"No valid rows in episode {episode_index}"

        stats = None
        for r in rows:
            a = np.array(r["action"], dtype=np.float32)
            if stats is None:
                stats = {"min": a.copy(), "max": a.copy(), "sum": a.copy(), "sumsq": a**2, "count": 1}
            else:
                stats["min"] = np.minimum(stats["min"], a)
                stats["max"] = np.maximum(stats["max"], a)
                stats["sum"] += a
                stats["sumsq"] += a**2
                stats["count"] += 1

        assert stats is not None, f"No valid actions in episode {episode_index}"
        stats = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}

        ds = Dataset.from_list(rows, features=self.features)
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        ds.to_parquet(str(tmp_path))
        os.replace(tmp_path, parquet_path)

        vid_chunk_dir = out_base.parent / "videos" / f"chunk-{episode_index // chunks_size:03d}" / "egocentric"
        vid_chunk_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_chunk_dir / f"episode_{episode_index:06d}.mp4"

        with iio.imopen(vid_path, "w", plugin="ffmpeg", fps=FPS, codec="libx264", macro_block_size=None) as f:
            for p in rgb_paths:
                f.write(iio.imread(p))

        return episode_index, len(rows), stats

    def run(self, data_root: Path, work_dir: Path, chunks_size: int, num_workers: int):
        self.chunks_size = chunks_size
        tdd = Path("task_description_dict.json")
        assert tdd.is_file(), "task_description_dict.json not found"
        self.task_description_dict = json.load(open(tdd))

        episode_sources: List[Tuple[int, Path, str, str]] = []
        task_index = 0
        self.tasks_meta = {}

        for task_name, task_dir, cat_name, leaf_name in iter_tasks(data_root):
            desc = self.task_description_dict.get(leaf_name, "")
            ep_dirs = [p for p in task_dir.iterdir() if p.is_dir() and re.match(r"episode_\d+", p.name)]
            ep_dirs = sorted(ep_dirs, key=lambda p: int(re.findall(r"\d+", p.name)[0]))
            for ep in ep_dirs:
                episode_sources.append((task_index, ep, desc, task_name))
            self.tasks_meta[task_index] = {"name": task_name, "category": cat_name, "description": desc}
            task_index += 1

        self.total_frames = 0
        total = len(episode_sources)
        self.global_stats = {"h1": None, "g1": None}

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.make_one_episode, task_idx, i, ep_dir, data_dir, chunks_size)
                for i, (task_idx, ep_dir, _task_meta, _task_name) in enumerate(episode_sources)
            ]
            data_stats: List[Tuple[int, int, Dict[str, Any]]] = []
            for fut in tqdm(as_completed(futures), total=total, desc="Processing episodes", unit="ep"):
                tmp_idx, n_frames, stats = fut.result()
                assert n_frames > 0, f"Episode {tmp_idx} has zero frames"
                data_stats.append((tmp_idx, n_frames, stats))


        data_stats.sort(key=lambda x: x[0])
        for ep_idx, n_frames, stats in data_stats:
            task_idx, ep_dir, task_dsc, task_name = episode_sources[ep_idx]
            self.kept_records.append((ep_idx, task_idx, ep_dir, task_dsc, task_name))
            self.lengths_by_episode[ep_idx] = n_frames
            self.num_episodes += 1
            self.total_frames += n_frames
            if stats:
                rtype = self.get_robot_type(ep_dir)
                cur = self.global_stats.get(rtype)
                if cur is None:
                    self.global_stats[rtype] = stats
                else:
                    if len(cur["min"]) == len(stats["min"]):
                        cur["min"] = np.minimum(np.array(cur["min"]), np.array(stats["min"])).tolist()
                        cur["max"] = np.maximum(np.array(cur["max"]), np.array(stats["max"])).tolist()
                        cur["sum"] = (np.array(cur["sum"]) + np.array(stats["sum"])).tolist()
                        cur["sumsq"] = (np.array(cur["sumsq"]) + np.array(stats["sumsq"])).tolist()
                        cur["count"] = int(cur["count"]) + int(stats["count"])

        print(f"\nKept {self.num_episodes}/{total} episodes, total {self.total_frames} frames")

    def write_meta(self, out_dir: Path):
        if self.num_episodes == 0:
            print("No valid episodes, aborting.")
            return

        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []

        for (ep_idx, task_idx, _ep_dir, task_dsc, _task_name) in sorted(self.kept_records, key=lambda x: x[0]):
            n = self.lengths_by_episode.get(ep_idx, 0)
            if n <= 0:
                continue
            ep_rows_meta.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [task_idx],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": self.get_robot_type(_ep_dir),
                    "instruction": task_dsc
                }
            )
            dataset_cursor += n

        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = []
        for ti, meta in self.tasks_meta.items():
            task_rows.append(
                {
                    "task_index": ti,
                    "task": meta.get("name", f"task_{ti:04d}"),
                    "category": meta.get("category", ""),
                    "description": meta.get("description", ""),
                }
            )
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.depth.egocentric": {"dtype": "float32", "shape": [480, 640], "names": ["height", "width"]},
            "observation.lidar": {"dtype": "float32", "shape": [-1, 3]},
            "observation.imu.quaternion": {"dtype": "float32", "shape": [4]},
            "observation.imu.accelerometer": {"dtype": "float32", "shape": [3]},
            "observation.imu.gyroscope": {"dtype": "float32", "shape": [3]},
            "observation.imu.rpy": {"dtype": "float32", "shape": [3]},
            "observation.odometry.position": {"dtype": "float32", "shape": [3]},
            "observation.odometry.velocity": {"dtype": "float32", "shape": [3]},
            "observation.odometry.rpy": {"dtype": "float32", "shape": [3]},
            "observation.odometry.quat": {"dtype": "float32", "shape": [4]},
            "observation.arm_joints": {"dtype": "float32", "shape": [-1]},
            "observation.leg_joints": {"dtype": "float32", "shape": [-1]},
            "observation.hand_joints": {"dtype": "float32", "shape": [-1]},
            "observation.tactile": {"dtype": "float32", "shape": [-1, -1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        robot_types = set(episodes_df["robot_type"].tolist()) if not episodes_df.empty else {"h1"}
        global_robot_type = list(robot_types)[0] if len(robot_types) == 1 else "mixed"

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type=global_robot_type,
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=FPS,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )

        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))

        with open(meta_dir / "tasks.jsonl", "w") as f_tasks:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f_tasks)
                f_tasks.write("\n")

        with open(meta_dir / "episodes.jsonl", "w") as f_eps:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f_eps)
                f_eps.write("\n")

        with open(meta_dir / "episodes_stats.jsonl", "w") as f_stats_eps:
            for row in episodes_df.to_dict(orient="records"):
                stats_obj = {"episode_index": row["episode_index"], "stats": {}}
                json.dump(stats_obj, f_stats_eps)
                f_stats_eps.write("\n")

        stats_out = {}
        for rtype, cur in self.global_stats.items():
            if cur:
                arr_sum = np.array(cur["sum"])
                arr_sumsq = np.array(cur["sumsq"])
                n = cur["count"]
                mean = (arr_sum / n).tolist()
                var = (arr_sumsq / n - np.square(arr_sum / n)).tolist()
                std = np.sqrt(var).tolist()
                stats_out[rtype] = {
                    "action_min": cur["min"],
                    "action_max": cur["max"],
                    "action_mean": mean,
                    "action_std": std,
                }

        (meta_dir / "stats.json").write_text(json.dumps(stats_out, indent=4))
        print(f"\nWrote meta (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, stats.json) and {self.num_episodes} episode(s) into: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--work-dir", type=str, default="_lerobot_build")
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--repo-exist-ok", action="store_true")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Max parallel workers (default: all CPUs)")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).resolve()
    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = HE2LeRobotConverter()
    pipeline.run(data_root, work_dir, args.chunks_size, args.num_workers)
    pipeline.write_meta(work_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set")
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=args.repo_exist_ok)
        upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(work_dir))
        create_tag(args.repo_id, tag="v" + CODE_VERSION, repo_type="dataset")
        print(f"\n✅ Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

