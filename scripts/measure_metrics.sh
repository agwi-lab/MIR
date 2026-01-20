cd ~/DRMNet
OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=4 \
python scripts/estimate.py \
  --input_img .data/drmnet_synth_test/9C4A0022-6d8fe2e88e/obj.exr \
  --input_normal .data/drmnet_synth_test/9C4A0022-6d8fe2e88e/normal.npy

# Check outputs/r0.exr exists, then:
python scripts/compute_metrics.py \
  .data/drmnet_synth_test/9C4A0022-6d8fe2e88e/r0.exr outputs/r0.exr
