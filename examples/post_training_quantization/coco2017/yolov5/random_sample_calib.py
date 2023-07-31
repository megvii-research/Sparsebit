import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--calib_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

with open(os.path.join(args.data_path, "train2017.txt"), "r") as f:
    img_paths = f.readlines()

calib_paths = random.sample(img_paths, args.calib_size)
with open(os.path.join(args.data_path, "calib2017.txt"), "w") as f:
    f.writelines(calib_paths)
