

# I need to convert the 21 bands into one .npy files

# data_set_path = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_extracted"
# # after this there is 450 to 650 - 21 bands
# # examples
# path_450 = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_extracted/450"
# path_650 = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_extracted/650"

# Inside this path_450 and all path_bands, there is three folders test, train and val.
# An inside of each one there is r_0.exr files the name of the files are the same for all bands, and all test train and test
# I need you to join this 21 bands into a .npy files
# so i get test, train, val folders and each one has .npy with all the bands inside of it


import OpenEXR
import Imath
import numpy as np
import os
from pathlib import Path

# 21 bands from 450 to 650 (step of 10)
bands = [str(b) for b in range(450, 651, 10)]  # 450, 460, ..., 650
splits = ["train", "test", "val"]

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def read_s0_channel(exr_path):
    exr = OpenEXR.InputFile(exr_path)
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    raw = exr.channel('s0', FLOAT)
    arr = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    return arr

# Get all filenames from the first band/split to know what files exist
def get_filenames(data_set_path,band, split):
    folder = os.path.join(data_set_path, band, split)
    return sorted([f for f in os.listdir(folder) if f.endswith('.exr')])

def split_npy_files(data_set_path,out_dir):
    for split in splits:
        save_dir = os.path.join(out_dir, split)

        first_band_split = os.path.join(data_set_path, bands[0], split)
        if not os.path.exists(first_band_split):
            print(f"The split {split} doesnt exits")
            continue
        os.makedirs(save_dir, exist_ok=True)
        # Get file list from first band
        
        filenames = get_filenames(data_set_path,bands[0], split)
        print(f"[{split}] Found {len(filenames)} files across {len(bands)} bands")

        for fname in filenames:
            out_name = fname.replace('.exr', '.npy')
            out_file = os.path.join(save_dir, out_name)
            if os.path.exists(out_file):
                print(f"Skipping already exits {out_file}")
                continue
            band_arrays = []
            for band in bands:
                exr_path = os.path.join(data_set_path, band, split, fname)
                arr = read_s0_channel(exr_path)
                band_arrays.append(arr)

            # Stack into (21, H, W)
            stacked = np.stack(band_arrays, axis=0)

            # Save as .npy
            
            
            np.save(out_file, stacked)
            print(f"  Saved {save_dir} -> shape {stacked.shape}")

    print("Done!")

# Using the .npy and using
from utils import spectrum_to_rgb
# in a new folder called hot_dog_rgb saved all the images from the .npy files converter from spectrum to rgb
# for this moment only use the train images, so create a new folder /train
import cv2

def npy2rgb(npy_dir,out_dir):

    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        saved_dir = os.path.join(out_dir, split)
        os.makedirs(saved_dir, exist_ok=True)
        
        split_npy_dir = os.path.join(npy_dir, split)
        if not os.path.exists(split_npy_dir):
            print(f"The split {split} doesnt exits")
            continue
        print(f"[{split}] Found {len(sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')]))} files")
        for fname in sorted(os.listdir(split_npy_dir)):
            if not fname.endswith('.npy'):
                continue
            out_name = fname.replace('.npy', '.png')
            out_file = os.path.join(out_dir,split,out_name)
            if os.path.exists(out_file):
                print(f"Skipping already exits {out_file}")
                continue
            # Load (21, H, W) -> (H, W, 21)
            data = np.load(os.path.join(split_npy_dir, fname))
            data = data.transpose(1, 2, 0).astype(np.float32)

            # Convert to RGB (H, W, 3), values in [0, 1]
            rgb = spectrum_to_rgb(data, start=450, end=650, bands=21, apply_gamma=True)

            # Save as PNG (convert to uint8, BGR for OpenCV)
            rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

            
            cv2.imwrite(os.path.join(saved_dir, out_name), bgr)
            print(f"Saved {out_name}")
        print(f"Done! split {split}")
    print("Done!")


import subprocess
def run_colmap(rgb_train, workspace):
    os.makedirs(os.path.join(workspace,"database"), exist_ok=True)
    os.makedirs(os.path.join(workspace,"dense"), exist_ok=True)
    os.makedirs(os.path.join(workspace,"sparse"), exist_ok=True)

    db = os.path.join(workspace,"database","database.db")

    def run(cmd):
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    run([
        "colmap", "feature_extractor",
        "--image_path", str(rgb_train),
        "--database_path", str(db),
        "--ImageReader.single_camera", "1"
    ])
    
    run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db)
    ])

    run([
        "colmap", "mapper",
        "--image_path", str(rgb_train),
        "--database_path", str(db),
        "--output_path", os.path.join(workspace, "sparse"),
        "--Mapper.multiple_models", "0"
    ])

    run([
        "colmap", "model_analyzer",
        "--path", os.path.join(workspace, "sparse/0")
    ])

def parse_args():
    
    return parser.parse_args()

if __name__ == "__main__":

    import argparse
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Dataset from exr to npy to rgb.")
    parser.add_argument("--dataset_exr", default="/")
    parser.add_argument("--dir_npy", default="/")
    parser.add_argument("--dir_rgb", default="/")
    parser.add_argument("--dir_colmap", default="/")
    
    args = parser.parse_args()

    print(args.dataset_exr)
    
    split_npy_files(args.dataset_exr, args.dir_npy)

    npy2rgb(args.dir_npy,args.dir_rgb)

    run_colmap(
        args.dir_rgb,
        args.dir_colmap
    )
