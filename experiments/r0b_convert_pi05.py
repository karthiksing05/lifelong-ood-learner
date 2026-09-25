"""Convert the JAX pi05_libero checkpoint to openpi's PyTorch format (model.safetensors + assets), once.
Runs in the openpi env on a GPU node:
    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r0b_convert_pi05.py" slurm/jobs/robot_gpu.sbatch
"""
import os
import runpy
import subprocess
import sys
import time

CACHE = "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints"
SRC, DST = f"{CACHE}/pi05_libero", f"{CACHE}/pi05_libero_pytorch"
t0 = time.time()
cmd = [sys.executable, "/coc/flash10/ksingara3/robot/openpi/examples/convert_jax_model_to_pytorch.py",
       "--checkpoint_dir", SRC, "--config_name", "pi05_libero", "--output_path", DST]
print("[r0b]", " ".join(cmd), flush=True)
rc = subprocess.call(cmd, cwd="/coc/flash10/ksingara3/robot/openpi")
print(f"[r0b] convert exit {rc} in {time.time() - t0:.0f}s", flush=True)
if rc == 0:
    for root, dirs, files in os.walk(DST):
        tot = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        print(f"[r0b] {root}: {len(files)} files {tot / 1e9:.2f} GB", flush=True)
    # the assets (norm stats) travel with the checkpoint; copy if the converter did not
    if not os.path.exists(os.path.join(DST, "assets")):
        subprocess.call(["cp", "-r", os.path.join(SRC, "assets"), os.path.join(DST, "assets")])
        print("[r0b] copied assets", flush=True)
sys.exit(rc)
