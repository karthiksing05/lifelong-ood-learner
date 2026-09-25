"""LIBERO demonstrations -> action chunks and observations in EXACTLY the units pi05_libero was trained on.

Source: `physical-intelligence/libero` (LeRobot v2.0 parquet; 40 tasks, 1693 episodes, 273,465 frames, 10 fps,
images inline as PNG, no videos).  openpi's data loader (training/data_loader.py) takes, at every frame i, the
next H = action_horizon actions with LeRobot's delta_timestamps, clamped to the episode (the last action repeats
past the end, `actions_is_pad` marks those steps), then Normalize with QUANTILES (pi0.5: x -> (x - q01) / (q99 -
q01 + 1e-6) * 2 - 1) and PadStatesAndActions to action_dim 32.  This module reproduces that arithmetic with
numpy only (no openpi import), so it runs in any env; `experiments/r1_policy_smoke.py` checks it against the
policy's own transforms.

Rows are flat [H * 7] float32 over the REAL action dims (70 for H = 10); `pad_rows` / `real_dims(H)` map to and
from the model's [H * 32] padded layout (PolicyBackbone does this itself).
"""
from __future__ import annotations

import dataclasses
import io
import json
import os

import numpy as np

DATA_ROOT = "/coc/flash10/ksingara3/robot/data/hf/physical-intelligence__libero"
NORM_STATS = ("/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero/assets/"
              "physical-intelligence/libero/norm_stats.json")
SUITES = {"libero_10": list(range(0, 10)), "libero_goal": list(range(10, 20)),
          "libero_object": list(range(20, 30)), "libero_spatial": list(range(30, 40))}
ACTION_DIM_REAL = 7
ACTION_DIM = 32
GRIPPER = 6                      # index of the gripper action (-1 open, +1 close in LIBERO)


def suite_of(task_index):
    for s, ids in SUITES.items():
        if int(task_index) in ids:
            return s
    return "unknown"


def real_dims(H, A=ACTION_DIM, a=ACTION_DIM_REAL):
    """Flat indices of the real action dims inside [H * A] rows."""
    return np.concatenate([np.arange(a) + h * A for h in range(H)])


# ── normalisation (openpi's quantile Normalize, then zero-pad to 32) ────────────────────────
@dataclasses.dataclass
class Norm:
    q01: np.ndarray
    q99: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, path=NORM_STATS, key="actions"):
        d = json.load(open(path))["norm_stats"][key]
        return cls(*(np.asarray(d[k], np.float32) for k in ("q01", "q99", "mean", "std")))

    def normalize(self, x):
        x = np.asarray(x, np.float32)
        q01, q99 = self.q01[: x.shape[-1]], self.q99[: x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize(self, x):
        x = np.asarray(x, np.float32)
        q01, q99 = self.q01[: x.shape[-1]], self.q99[: x.shape[-1]]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def to_rows(chunks7, norm):
    """[n, H, 7] raw actions -> flat normalised rows [n, H * 7]."""
    c = np.asarray(chunks7, np.float32)
    n, H, a = c.shape
    return norm.normalize(c).reshape(n, H * a)


def from_rows(rows, norm, H, a=ACTION_DIM_REAL):
    """flat rows [n, H * 7] -> raw actions [n, H, 7]."""
    return norm.unnormalize(np.asarray(rows, np.float32).reshape(-1, H, a))


def pad_rows(rows, H, a=ACTION_DIM_REAL, A=ACTION_DIM):
    """flat real rows [n, H * 7] -> the model's zero-padded [n, H, A]."""
    r = np.asarray(rows, np.float32).reshape(-1, H, a)
    out = np.zeros((len(r), H, A), np.float32)
    out[:, :, :a] = r
    return out


# ── the dataset ──────────────────────────────────────────────────────────────────────────────
def chunk_indices(T, H):
    """LeRobot's delta-timestamp indices for every frame: [T, H] clamped to the episode, and the pad mask."""
    idx = np.arange(T)[:, None] + np.arange(H)[None, :]
    pad = idx >= T
    return np.minimum(idx, T - 1), pad


class Demos:
    """The LeRobot metadata plus lazy parquet access.  `episodes(task)` -> list of episode indices."""

    def __init__(self, root=DATA_ROOT):
        self.root = root
        self.info = json.load(open(os.path.join(root, "meta", "info.json")))
        self.tasks = {int(j["task_index"]): j["task"] for j in map(json.loads, open(os.path.join(root, "meta", "tasks.jsonl")))}
        self.task_of_text = {v: k for k, v in self.tasks.items()}
        self.eps = [json.loads(l) for l in open(os.path.join(root, "meta", "episodes.jsonl"))]
        self.fps = float(self.info["fps"])
        self.chunks_size = int(self.info["chunks_size"])
        self._cache = {}

    def episodes(self, task):
        """Episode indices of one task (int index or suite name)."""
        if isinstance(task, str):
            ids = set(SUITES[task])
        else:
            ids = {int(task)}
        return [e["episode_index"] for e in self.eps if self.task_of_text[e["tasks"][0]] in ids]

    def path(self, episode_index):
        return os.path.join(self.root, self.info["data_path"].format(episode_chunk=episode_index // self.chunks_size,
                                                                     episode_index=episode_index))

    def load(self, episode_index, images=False):
        """-> dict(state [T, 8], actions [T, 7], task_index, task, frame_index, (image, wrist_image: PNG bytes))."""
        import pyarrow.parquet as pq
        cols = ["state", "actions", "frame_index", "task_index"] + (["image", "wrist_image"] if images else [])
        t = pq.read_table(self.path(episode_index), columns=cols)
        d = t.to_pydict()
        out = dict(state=np.asarray(d["state"], np.float32), actions=np.asarray(d["actions"], np.float32),
                   frame_index=np.asarray(d["frame_index"]), task_index=int(d["task_index"][0]),
                   episode_index=int(episode_index))
        out["task"] = self.tasks[out["task_index"]]
        if images:
            out["image"] = [x["bytes"] for x in d["image"]]
            out["wrist_image"] = [x["bytes"] for x in d["wrist_image"]]
        return out

    @staticmethod
    def decode(png_bytes):
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), np.uint8)

    def observation(self, ep, i):
        """The raw observation dict at frame i (for PolicyBackbone.observation): images 256x256 uint8 HWC as
        stored (openpi resizes to 224 inside its transforms), state 8-D raw, prompt = the task text."""
        if "image" not in ep:
            ep2 = self.load(ep["episode_index"], images=True)
            ep.update(image=ep2["image"], wrist_image=ep2["wrist_image"])
        return {"observation/image": self.decode(ep["image"][i]), "observation/wrist_image": self.decode(ep["wrist_image"][i]),
                "observation/state": ep["state"][i], "prompt": ep["task"]}


def gripper_events(actions):
    """Frames where the gripper command changes sign (grasp / release).  -> int array."""
    g = np.sign(np.asarray(actions)[:, GRIPPER])
    return np.where(np.diff(g) != 0)[0] + 1


@dataclasses.dataclass
class ChunkSet:
    """Every chunk of a set of episodes, as rows, with its provenance."""
    rows: np.ndarray            # [N, H * 7] normalised (real dims)
    raw: np.ndarray             # [N, H, 7]
    task: np.ndarray            # [N]
    episode: np.ndarray         # [N]
    frame: np.ndarray           # [N]  start frame in the episode
    pad: np.ndarray             # [N]  number of padded steps at the end (0 for interior chunks)
    H: int

    def __len__(self):
        return len(self.rows)

    def select(self, m):
        return ChunkSet(self.rows[m], self.raw[m], self.task[m], self.episode[m], self.frame[m], self.pad[m], self.H)


def chunkset(demos, episodes, H=10, norm=None, stride=1, drop_padded=False):
    """Chunks of `episodes` at every frame (or every `stride`-th).  -> ChunkSet."""
    norm = norm or Norm.load()
    rows, raw, task, epi, frame, pad = [], [], [], [], [], []
    for e in episodes:
        ep = demos.load(e)
        T = len(ep["actions"])
        idx, pm = chunk_indices(T, H)
        sel = np.arange(0, T, stride)
        c = ep["actions"][idx[sel]]                       # [n, H, 7]
        npad = pm[sel].sum(1)
        if drop_padded:
            keep = npad == 0
            sel, c, npad = sel[keep], c[keep], npad[keep]
        raw.append(c)
        rows.append(to_rows(c, norm))
        task.append(np.full(len(sel), ep["task_index"]))
        epi.append(np.full(len(sel), e))
        frame.append(sel)
        pad.append(npad)
    cat = lambda xs, dt=None: np.concatenate(xs).astype(dt) if dt else np.concatenate(xs)   # noqa: E731
    return ChunkSet(cat(rows, np.float32), cat(raw, np.float32), cat(task, int), cat(epi, int), cat(frame, int), cat(pad, int), H)


# ── LIBERO-90 (held out from pi05_libero's fine-tune): IPEC-COMMUNITY/libero_90_no_noops_lerobot ─────────────
L90_REPO = "IPEC-COMMUNITY/libero_90_no_noops_lerobot"
L90_ROOT = "/coc/flash10/ksingara3/robot/data/hf/IPEC-COMMUNITY__libero_90_no_noops_lerobot"


class Demos90(Demos):
    """LIBERO-90 demos in the IPEC LeRobot v2.1 port, mapped to pi05_libero's conventions.

    Differences from physical-intelligence/libero, all verified on 2026-09-25: columns `action` /
    `observation.state` (same 8-D state: eef pos, axis-angle, finger qpos), images as mp4 videos, tasks MERGED by
    language across scenes (73 texts for 90 benchmark tasks: use tasks whose text is unique), and the gripper
    action is 1 = open / 0 = close where pi0.5's data uses -1 = open / +1 = close: g_pi = 1 - 2 g_ipec.
    Parquet files are fetched on demand (hf_hub_download, one file per episode)."""

    def __init__(self, root=L90_ROOT):
        super().__init__(root)

    def fetch(self, episode_index):
        p = self.path(episode_index)
        if not os.path.exists(p):
            from huggingface_hub import hf_hub_download
            rel = os.path.relpath(p, self.root)
            hf_hub_download(L90_REPO, rel, repo_type="dataset", local_dir=self.root)
        return p

    def load(self, episode_index, images=False):
        import pyarrow.parquet as pq
        if images:
            raise NotImplementedError("LIBERO-90 frames are mp4 videos; decode them separately")
        t = pq.read_table(self.fetch(episode_index), columns=["observation.state", "action", "frame_index", "task_index"]).to_pydict()
        act = np.asarray(t["action"], np.float32)
        act[:, GRIPPER] = 1.0 - 2.0 * act[:, GRIPPER]
        out = dict(state=np.asarray(t["observation.state"], np.float32), actions=act, frame_index=np.asarray(t["frame_index"]),
                   task_index=int(t["task_index"][0]), episode_index=int(episode_index))
        out["task"] = self.tasks[out["task_index"]]
        return out

    def episodes_of_text(self, text):
        return [e["episode_index"] for e in self.eps if e["tasks"][0] == text]

    def frames(self, episode_index):
        """Decode both camera streams of an episode (AV1 via PyAV's libdav1d).  -> (image [T,256,256,3], wrist [T,...]).
        Same orientation as pi05_libero's training frames (checked against LIBERO-10 KITCHEN_SCENE3, 2026-09-25)."""
        import av
        from huggingface_hub import hf_hub_download
        out = []
        for key in ("observation.images.image", "observation.images.wrist_image"):
            rel = self.info["video_path"].format(episode_chunk=episode_index // self.chunks_size, video_key=key, episode_index=episode_index)
            p = os.path.join(self.root, rel)
            if not os.path.exists(p):
                hf_hub_download(L90_REPO, rel, repo_type="dataset", local_dir=self.root)
            with av.open(p) as c:
                out.append(np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]))
        return out[0], out[1]

    def observation(self, ep, i):
        if "image" not in ep:
            ep["image"], ep["wrist_image"] = self.frames(ep["episode_index"])
        return {"observation/image": ep["image"][i], "observation/wrist_image": ep["wrist_image"][i],
                "observation/state": ep["state"][i], "prompt": ep["task"]}
