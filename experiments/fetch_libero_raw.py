"""Download LIBERO-90's raw hdf5 demonstrations (full simulator state per step) for the given tasks, for experiments/p0_demo_init.py.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/fetch_libero_raw.py 8+9+10
"""
import re
import sys

from huggingface_hub import hf_hub_download

TASKMAP = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
OUT = "/coc/flash10/ksingara3/robot/data/libero_raw"


def main():
    src = open(TASKMAP).read()
    block = src[src.index('"libero_90"'):]
    block = block[block.index("[") + 1:block.index("]")]
    names = re.findall(r'"([^"]+)"', block)
    for t in [int(x) for x in sys.argv[1].split("+")]:
        p = hf_hub_download("yifengzhu-hf/LIBERO-datasets", f"libero_90/{names[t]}_demo.hdf5", repo_type="dataset", local_dir=OUT)
        print(t, p, flush=True)


if __name__ == "__main__":
    main()
