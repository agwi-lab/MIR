import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
sys.path.append('/home/buka2004/DRMNet/')
import torch
from utils.file_io import load_exr, load_png, save_png
from pathlib import Path

exr_path = Path('/home/buka2004/DRMNet/.data/drmnet_synth_test/9C4A0022-6d8fe2e88e/obj.exr')
input_img = load_exr(exr_path, as_torch=True)
input_img
