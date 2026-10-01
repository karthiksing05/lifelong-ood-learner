"""Media for the results site (experiments/site_page.py): side-by-side rollout videos on the same init state, prototype clips from
the demos, scene stills.  Every clip is cut from recorded episodes; nothing is rendered anew.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/site_media.py      (CPU; ffmpeg from the LIBERO venv)
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "site", "media")
FF = "/coc/flash10/ksingara3/robot/openpi/examples/libero/.venv/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2"
FONTS = os.path.expanduser("~/flash/miniconda3/envs/chunks/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf")
F_B = lambda s: ImageFont.truetype(os.path.join(FONTS, "DejaVuSans-Bold.ttf"), s)
F_R = lambda s: ImageFont.truetype(os.path.join(FONTS, "DejaVuSans.ttf"), s)
GOOD, BAD, INK, PAPER = (19, 138, 96), (190, 62, 48), (20, 24, 22), (246, 247, 245)


def read_mp4(path, size=224):
    raw = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def write_mp4(frames, path, fps=20, crf=27):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow", "-movflags", "+faststart", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    return os.path.getsize(path)


def episode(run, t, k):
    """(frames, success) of trial k of a run directory's saved video."""
    vs = glob.glob(os.path.join(R, run, f"task{t}_trial{k}_*.mp4"))
    if not vs:
        return None, None
    return read_mp4(vs[0]), "success" in vs[0]


def panel(frame, title, sub, status, scale=2):
    """One labelled panel: the frame upscaled, a title bar above, a status strip below."""
    img = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.BICUBIC)
    W, H = img.size
    top, bot = 50, 34
    can = Image.new("RGB", (W, H + top + bot), PAPER)
    can.paste(img, (0, top))
    d = ImageDraw.Draw(can)
    d.text((10, 6), title, font=F_B(19), fill=INK)
    d.text((10, 29), sub, font=F_R(14), fill=(88, 98, 94))
    if status is not None:
        txt, col = status
        d.rectangle([0, H + top, W, H + top + bot], fill=col)
        d.text((10, H + top + 7), txt, font=F_B(16), fill=(255, 255, 255))
    else:
        d.rectangle([0, H + top, W, H + top + bot], fill=(221, 226, 223))
    return np.asarray(can)


def side_by_side(specs, out, t, k, title_hold=0):
    """specs = [(run dir, title, sub)] on task t, trial k (the same init state): panels play together, each holding its last frame;
    the status strip shows the outcome once that episode ends."""
    eps = [(episode(run, t, k), ti, su) for run, ti, su in specs]
    if any(e[0][0] is None for e in eps):
        print("missing", out, [run for run, _, _ in specs])
        return None
    n = max(len(e[0][0]) for e in eps)
    frames = []
    for i in range(n):
        cols = []
        for (fr, ok), ti, su in eps:
            j = min(i, len(fr) - 1)
            ended = i >= len(fr) - 1
            status = None
            if ended:
                status = (f"success at step {len(fr)}" if ok else "failed: 400-step limit", GOOD if ok else BAD)
            cols.append(panel(fr[j], ti, su, status))
            cols.append(np.full((cols[-1].shape[0], 8, 3), 246, np.uint8))
        frames.append(np.concatenate(cols[:-1], axis=1))
    frames += [frames[-1]] * 20                                   # hold the verdicts for a second
    size = write_mp4(frames, os.path.join(OUT, out))
    print(f"{out}: {len(frames)} frames, {size / 1e3:.0f} KB", flush=True)
    return dict(file=out, task=t, trial=k, outcomes=[bool(e[0][1]) for e in eps], frames=len(frames))


def side_by_side_tasks(specs, out, k):
    """Like side_by_side, but each panel from its own task: specs = [(run dir, task, title, sub)] at init state k."""
    eps = [(episode(run, t, k), ti, su) for run, t, ti, su in specs]
    if any(e[0][0] is None for e in eps):
        print("missing", out, [run for run, _, _, _ in specs])
        return None
    n = max(len(e[0][0]) for e in eps)
    frames = []
    for i in range(n):
        cols = []
        for (fr, ok), ti, su in eps:
            j = min(i, len(fr) - 1)
            status = (f"success at step {len(fr)}" if ok else "failed: 400-step limit", GOOD if ok else BAD) if i >= len(fr) - 1 else None
            cols.append(panel(fr[j], ti, su, status))
            cols.append(np.full((cols[-1].shape[0], 8, 3), 246, np.uint8))
        frames.append(np.concatenate(cols[:-1], axis=1))
    frames += [frames[-1]] * 20
    size = write_mp4(frames, os.path.join(OUT, out))
    print(f"{out}: {len(frames)} frames, {size / 1e3:.0f} KB", flush=True)
    return dict(file=out, task=specs[-1][1], trial=k, outcomes=[bool(e[0][1]) for e in eps], frames=len(frames))


VERBS = [("open move", "move to the object, gripper open"), ("open reach down", "reach down onto it"),
         ("open grip change", "close the gripper on it"), ("closed lift", "lift it"),
         ("closed transport", "carry it"), ("closed lower", "lower it into place")]


def task_names():
    import re
    src = open("/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py").read()
    block = src[src.index('"libero_90"'):]
    return re.findall(r'"([^"]+)"', block[block.index("[") + 1:block.index("]")])


def demo_frames(t, names):
    """Task t's ONE demo as the raw recording's agentview frames (128 x 128, upright), one per kept demo step."""
    import h5py
    d = np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)
    m = np.load(os.path.join(R, "p0_demo", f"{t}_match.npz"), allow_pickle=True)
    with h5py.File(f"/coc/flash10/ksingara3/robot/data/libero_raw/libero_90/{names[t]}_demo.hdf5", "r") as h:
        rgb = h["data"][str(d["raw_key"])]["obs"]["agentview_rgb"][:]
    return rgb[np.asarray(m["kept"])][:, ::-1, ::-1]


def prototype_clips(lang):
    """Per verb (the t = 0.4 prototypes named alike, experiments/grammar_readout.py), the demo frames where it is active in four tasks
    from different scenes, as a looping 2 x 2 grid."""
    L = np.load(os.path.join(R, "grammar_labels.npz"))
    g = json.load(open(os.path.join(R, "grammar_readout.json")))
    names = task_names()
    scene = {t: n.split("_SCENE")[0] + "_SCENE" + n.split("_SCENE")[1].split("_")[0] for t, n in enumerate(names)}
    tasks, cut = [int(t) for t in L["tasks"]], L["cut"]
    verb_of = {int(c): v["name"].split(" (")[0] for c, v in g["prototypes"]["0.4"].items()}
    out = []
    cache = {}
    for verb, gloss in VERBS:
        ids = sorted(c for c, n in verb_of.items() if n == verb)
        runs = []
        for k, t in enumerate(tasks):
            lab = [verb_of.get(int(x)) for x in L["lab_0.4"][cut[k]:cut[k + 1]]]
            best, i = None, 0
            while i < len(lab):
                if lab[i] == verb:
                    j = i
                    while j + 1 < len(lab) and lab[j + 1] == verb:
                        j += 1
                    if best is None or j - i > best[1] - best[0]:
                        best = (i, j)
                    i = j + 1
                else:
                    i += 1
            if best:
                runs.append((best[1] - best[0] + 1, t, best))
        runs.sort(reverse=True)
        pick, seen = [], set()
        for n, t, (a, b) in runs:
            if scene[t] in seen:
                continue
            pick.append((t, a, b))
            seen.add(scene[t])
            if len(pick) == 4:
                break
        clips = []
        for t, a, b in pick:
            if t not in cache:
                cache[t] = demo_frames(t, names)
            fr = cache[t]
            pad = 8 if b - a < 6 else 4                           # one- or two-frame verbs (the grip change): more context around them
            lo, hi = max(0, a - pad), min(len(fr), b + pad + 1)
            if hi - lo > 44:                                       # long runs (transport): 4 s around their middle
                mid = (lo + hi) // 2
                lo, hi = mid - 22, mid + 22
            clips.append((t, fr[lo:hi], a - lo, b - lo))
        n = max(len(c[1]) for c in clips)
        frames = []
        for i in range(n):
            cells = []
            for t, fr, a, b in clips:
                j = min(i, len(fr) - 1)
                img = Image.fromarray(fr[j]).resize((224, 224), Image.BICUBIC)
                can = Image.new("RGB", (224, 224 + 40), PAPER)
                can.paste(img, (0, 0))
                d = ImageDraw.Draw(can)
                on = a <= j <= b
                d.rectangle([0, 224, 224, 229], fill=GOOD if on else (221, 226, 223))
                txt = (lang.get(t) or names[t].split("_SCENE")[1].split("_", 1)[1].replace("_", " "))
                d.text((6, 234), (txt[:30] + "…") if len(txt) > 31 else txt, font=F_R(12), fill=INK)
                cells.append(np.asarray(can))
            row1 = np.concatenate([cells[0], np.full((264, 6, 3), 246, np.uint8), cells[1]], axis=1)
            row2 = np.concatenate([cells[2], np.full((264, 6, 3), 246, np.uint8), cells[3]], axis=1) if len(cells) > 3 else np.full_like(row1, 246)
            frames.append(np.concatenate([row1, np.full((6, row1.shape[1], 3), 246, np.uint8), row2], axis=0))
        fn = "proto_" + verb.replace(" ", "_") + ".mp4"
        size = write_mp4(frames, os.path.join(OUT, fn), fps=6, crf=24)
        tasks_with = sorted({t for _, t, _ in runs})
        out.append(dict(verb=verb, gloss=gloss, file=fn, prototype_ids=ids, tasks=[c[0] for c in clips], n_tasks=len(tasks_with),
                        frames=len(frames)))
        print(f"{fn}: {len(frames)} frames, {size / 1e3:.0f} KB, tasks {[c[0] for c in clips]}", flush=True)
    return out


def stills(lang):
    """Scene mosaic (the first frame of each of the 50 stream tasks at init state 30) and the walkthrough's frames (task 48: its demo
    at the phase events; our rollout at a new init state)."""
    names = task_names()
    pool = sorted(int(t) for t in json.load(open(os.path.join(R, "p5_pool.json"))))
    thumbs = []
    for t in pool:
        vs = sorted(glob.glob(os.path.join(R, f"m3_final_{t}_phasebind2_rjpred", f"task{t}_trial30_*.mp4")))
        if vs:
            thumbs.append(Image.fromarray(read_mp4(vs[0])[0]).resize((160, 160), Image.BICUBIC))
    cols = 10
    rows = (len(thumbs) + cols - 1) // cols
    mos = Image.new("RGB", (cols * 164 - 4, rows * 164 - 4), PAPER)
    for i, im in enumerate(thumbs):
        mos.paste(im, ((i % cols) * 164, (i // cols) * 164))
    mos.save(os.path.join(OUT, "mosaic.jpg"), quality=84)
    demo = demo_frames(48, names)
    keys = {"start": 0, "reach": 70, "grasp": 80, "lift": 100, "carry": 140, "release": 166}
    for k, i in keys.items():
        Image.fromarray(demo[min(i, len(demo) - 1)]).resize((256, 256), Image.BICUBIC).save(os.path.join(OUT, f"demo48_{k}.jpg"), quality=88)
    roll = read_mp4(sorted(glob.glob(os.path.join(R, "m3_final_48_phasebind2_rjpred_pca512_rj", "task48_trial30_*.mp4")))[0])
    n = len(roll)
    for k, f in {"0": 0, "1": int(n * .35), "2": int(n * .55), "3": int(n * .8), "4": n - 1}.items():
        Image.fromarray(roll[f]).save(os.path.join(OUT, f"roll48_{k}.jpg"), quality=88)
    print(f"stills: mosaic of {len(thumbs)} tasks, demo 48 ({len(demo)} frames), rollout 48 ({n} frames)", flush=True)
    return dict(mosaic_tasks=len(thumbs), demo48_frames=len(demo), demo48_keys=keys, roll48_frames=n)


def hand_trace(e, j, h=300, w=150):
    """The hand's path so far, drawn the way the camera sees the table: from above (left = image left, forward = image down) and
    from the side (forward to the right, up = up); one scale per clip, with a scale bar."""
    d = (e - e[0]) * 100                                                   # cm, robot frame: x forward, y left, z up
    s = max(5.0, float(np.abs(d).max()) * 1.15)                             # half-range of both views, cm
    can = Image.new("RGB", (w, h), PAPER)
    dr = ImageDraw.Draw(can)
    box = 112
    views = [("from above", 16, lambda p: (-p[1], p[0])), ("from the side", 16 + box + 20, lambda p: (p[0], -p[2]))]
    for name, top, f in views:
        x0, y0 = 12, top
        dr.rectangle([x0, y0, x0 + box, y0 + box], outline=(205, 210, 206), fill=(255, 255, 255))
        dr.text((x0, y0 - 15), name, font=F_R(11), fill=(88, 98, 94))
        px = lambda p: (x0 + box / 2 + f(p)[0] / s * box / 2, y0 + box / 2 + f(p)[1] / s * box / 2)
        pts = [px(p) for p in d[: j + 1]]
        if len(pts) > 1:
            dr.line(pts, fill=GOOD, width=3)
        cx, cy = pts[0]
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=INK, width=2)
        cx, cy = pts[-1]
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=GOOD)
    bar = 10 if s > 12 else 2                                               # scale bar, cm
    L = bar / s * box / 2
    y = 16 + 2 * box + 20 + 18
    dr.line([12, y, 12 + L, y], fill=INK, width=2)
    dr.text((16 + L, y - 7), f"{bar} cm", font=F_R(11), fill=INK)
    return np.asarray(can)


def proto_exec_video():
    """The executed prototypes (experiments/proto_exec.py): a 3 x 2 grid, each verb's frame beside the hand's path so far (the large
    moves carry the hand out of the camera's view), labelled with the hand's displacement."""
    z = np.load(os.path.join(R, "proto_exec.npz"), allow_pickle=True)
    verbs = [k[len("frames_"):] for k in z.files if k.startswith("frames_")]
    n = max(len(z[f"frames_{v}"]) for v in verbs)
    moves = []
    frames = []
    for i in range(n):
        cells = []
        for v in verbs:
            fr = z[f"frames_{v}"]
            e = z[f"eef_{v}"]
            j = min(i, len(fr) - 1)
            d = (e[j] - e[0]) * 100
            img = Image.fromarray(fr[j]).resize((256, 256), Image.BICUBIC)
            can = Image.new("RGB", (256 + 150, 256 + 44), PAPER)
            can.paste(img, (0, 0))
            can.paste(Image.fromarray(hand_trace(e, j)), (256, 0))
            dr = ImageDraw.Draw(can)
            dr.text((8, 260), v.replace("_", " "), font=F_B(15), fill=INK)
            dr.text((8, 279), f"hand {d[0]:+.0f} fwd {d[1]:+.0f} left {d[2]:+.0f} up (cm), step {j}", font=F_R(12), fill=(88, 98, 94))
            cells.append(np.asarray(can))
        sep = np.full((300, 6, 3), 246, np.uint8)
        r1 = np.concatenate([cells[0], sep, cells[1], sep, cells[2]], axis=1)
        r2 = np.concatenate([cells[3], sep, cells[4], sep, cells[5]], axis=1)
        frames.append(np.concatenate([r1, np.full((6, r1.shape[1], 3), 246, np.uint8), r2], axis=0))
    frames += [frames[-1]] * 10
    for v in verbs:
        e = z[f"eef_{v}"]
        d = (e[-1] - e[0]) * 100
        moves.append(f"{v.replace('_', ' ')}: {d[0]:+.0f} forward, {d[1]:+.0f} left, {d[2]:+.0f} up (cm)")
    size = write_mp4(frames, os.path.join(OUT, "proto_exec.mp4"), fps=10, crf=24)
    print(f"proto_exec.mp4: {len(frames)} frames, {size / 1e3:.0f} KB", flush=True)
    return dict(file="proto_exec.mp4", language=str(z["language"]), moves="Net hand displacement after four repetitions: " + "; ".join(moves) + ".")


def posters():
    """A poster per clip: the last frame of a comparison (it shows every panel's outcome), the middle frame of a prototype clip."""
    for f in sorted(glob.glob(os.path.join(OUT, "*.mp4"))):
        at = ["-sseof", "-0.2"] if not os.path.basename(f).startswith("proto_") or "exec" in f else ["-ss", "1.0"]
        subprocess.run([FF, "-v", "error", "-y"] + at + ["-i", f, "-frames:v", "1", "-q:v", "4", f[:-4] + ".jpg"], check=True)
    print("posters:", len(glob.glob(os.path.join(OUT, "*.mp4"))), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    lang = {int(t): v["language"] for t, v in pool.items()}
    meta = {"comparisons": [], "compositions": []}
    BASE = [("m3_final_{t}_zeroshot", "pi0.5, zero-shot", "no adaptation"),
            ("m3_final_{t}_lora_r17bal", "LoRA, best fine-tuning", "demo + practice, 18.7 MB"),
            ("m3_final_{t}_phasebind2_rjpred_pca512_rj", "Bindings (ours)", "frozen pi0.5 + 9.6 KB")]
    only = sys.argv[1:] or ["cmp", "comp", "stills", "proto"]
    if "cmp" not in only:
        meta = json.load(open(os.path.join(OUT, "media.json"))) if os.path.exists(os.path.join(OUT, "media.json")) else meta
    for t, k in (((35, 30), (42, 30), (32, 31), (1, 30), (34, 30), (5, 31), (37, 31), (39, 30)) if "cmp" in only else ()):
        r = side_by_side([(a.format(t=t), b, c) for a, b, c in BASE], f"cmp_{t}_{k}.mp4", t, k)
        if r:
            r["language"] = lang.get(t)
            meta["comparisons"].append(r)
    COMP = [("fix_3", 3, [("dev_tdiag_3_streamt_4_4", "Transport, on the clock", "waypoints advance by time"),
                          ("dev_tdiag2_3_streamte_4_4", "Transport, on arrival", "waypoints passed when reached")]),
            ("fix_48", 48, [("dev_tdiag_48_streamt_58_53", "Transport, on the clock", "waypoints advance by time"),
                            ("dev_tdiag2_48_streamte_58_53", "Transport, on arrival", "waypoints passed when reached")]),
            ("carries_16", 16, [("dev_p5_16_streamc_1_17", "Source's route", "the place demo's path"),
                                ("dev_handoff_16_streamh_1_17", "Prior carries", "pi0.5 moves the object"),
                                ("dev_carry_16_streamte_1_17", "Lift, cross, descend", "on arrival")]),
            ("carries_13", 13, [("dev_p5_13_streamc_17_65", "Source's route", "the place demo's path"),
                                ("dev_handoff_13_streamh_17_65", "Prior carries", "pi0.5 moves the object"),
                                ("dev_carry_13_streamte_17_65", "Lift, cross, descend", "on arrival")])]
    if "comp" in only:
        meta["compositions"] = []
        for name, t, specs in COMP:
            r = side_by_side(specs, f"comp_{name}.mp4", t, 0)
            if r:
                r.update(language=lang.get(t), name=name)
                meta["compositions"].append(r)
    if "stills" in only:
        meta["stills"] = stills(lang)
    if "proto" in only:
        meta["prototypes"] = prototype_clips(lang)
    if "overhaul" in only:                                   # the overhaul's figures (Read -> Compose -> Fold)
        OV = [("ov_habit_48", 48, 0, [("f1_bind_46_zeroshot", "Asked for the alphabet soup", "pi0.5, living room 1"),
                                      ("f1_bind_48_zeroshot", "Asked for the ketchup", "pi0.5, the same scene")], 46),
              ("ov_48_30", 48, 30, [("m3_final_48_zeroshot", "pi0.5, zero-shot", "the instruction alone"),
                                    ("m3_final_48_fcomp_8", "Composition", "own conditionals, weight 8"),
                                    ("m3_final_48_ftoken_self", "Factor token", "18.4 KB, policy frozen")], None),
              ("ov_58_0", 58, 0, [("f1_bind_58_zeroshot", "pi0.5, zero-shot", "ketchup into the tray"),
                                  ("f4_pilot_58_ftoken_48_32", "Ketchup token", "fitted on two other tasks")], None),
              ("ov_3_31", 3, 31, [("m3_final_3_ftoken_self", "Factor token", "18.4 KB, policy frozen"),
                                  ("m3_final_3_lora", "LoRA", "18.7 MB, one demo")], None)]
        meta["overhaul"] = {}
        for name, t, k, specs, t_first in OV:
            if t_first is not None:                       # panels from two tasks of one scene, at the same init state
                eps = []
                r = side_by_side_tasks([(run, tt, ti, su) for (run, ti, su), tt in zip(specs, (t_first, t))], f"{name}.mp4", k)
            else:
                r = side_by_side(specs, f"{name}.mp4", t, k)
            if r:
                r["language"] = lang.get(t)
                meta["overhaul"][name] = r
    if "exec" in only:
        meta["proto_exec"] = proto_exec_video()
    json.dump(meta, open(os.path.join(OUT, "media.json"), "w"), indent=1)
    if "posters" in only or len(only) > 1:
        posters()


if __name__ == "__main__":
    main()
