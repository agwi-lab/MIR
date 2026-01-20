"""DRMNet Batch Validation - SSIM/PSNR Metrics on drmnet_synth_test dataset"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt


# Existing imports (keep all)
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")


import cv2


sys.path.append(str(Path(__file__).parent.parent))


from dataset.basedataset import BaseDataset
from ldm.util import instantiate_from_config
from models.drmnet import DRMNet
from models.obsnet import ObsNetDiffusion
from utils.file_io import load_exr, load_png, save_png
from utils.img2refmap import refmap_mask_make
from utils.mitsuba3_utils import get_bsdf, visualize_bsdf
from utils.tonemap import hdr2ldr


def save_exr(path: Path, img: np.ndarray):
    """Save numpy array as EXR using standard OpenCV constants"""
    # Ensure float32, HWC, 3 channels
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Convert CHW to HWC if needed
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Ensure 3 channels (HWC)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] != 3:
        img = img[:, :, :3]
    
    # Standard OpenCV EXR - NO scaling to 255, FLOAT type
    success = cv2.imwrite(
        str(path), 
        img, 
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    )
    
    if not success:
        # Fallback: save as numpy for debugging
        np.save(str(path.with_suffix('.npy')), img)
        print(f"WARNING: EXR save failed for {path}, saved as NPY")
    else:
        print(f"Saved EXR: {path}")


def calculate_psnr(img1, img2, max_value=9128): # HDR data diaposon
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def compute_metrics(gt_path: Path, pred_path: Path, input_img_path: Path) -> dict:
    """Compute SSIM/PSNR/RMSE using direct OpenCV loading + Imax from input"""
    try:
        # Load GT
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            print(f"Cannot load GT: {gt_path}")
            return None
        gt_img = cv2.cvtColor(gt_raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Load predicted
        pred_raw = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred_raw is None:
            print(f"Cannot load PRED: {pred_path}")
            return None
        pred_img = cv2.cvtColor(pred_raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Load input EXR for Imax
        input_raw = cv2.imread(str(input_img_path), cv2.IMREAD_UNCHANGED)
        input_img = cv2.cvtColor(input_raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        Imax = 9128

        # Compute metrics
        mse = np.mean((gt_img - pred_img) ** 2)
        rmse = float(np.sqrt(mse))
        psnr_val = calculate_psnr(gt_img, pred_img, max_value=Imax)
        ssim_val = ssim(gt_img, pred_img, multichannel=True, data_range=1.0, channel_axis=-1)

        return {
            'rmse': rmse,
            'ssim': float(ssim_val),
            'psnr': float(psnr_val),
            'Imax': Imax,
            'gt_shape': str(gt_img.shape),
            'pred_shape': str(pred_img.shape),
            'gt_path': str(gt_path),
            'pred_path': str(pred_path),
            'input_path': str(input_img_path)
        }
    except Exception as e:
        print(f"Metrics error {gt_path}: {e}")
        return None


# Keep ORIGINAL estimate() function EXACTLY unchanged
def estimate(
    DRMNet_model: DRMNet,
    ObsNet_model: ObsNetDiffusion,
    input_img: torch.Tensor,
    input_normal: torch.Tensor,
    mask: torch.Tensor,
    tag: str = "sample",
    erode_kernel_size: int = 5,
):
    refmap_res = DRMNet_model.ds.size
    torch.cuda.synchronize()

    if erode_kernel_size > 0:
        inv_mask = ~mask
        kernel = torch.stack(torch.meshgrid(*torch.arange(erode_kernel_size, device="cuda").expand(2, -1), indexing="ij"))
        kernel = kernel + 0.5
        kernel = torch.linalg.norm(kernel - erode_kernel_size / 2, axis=0) <= erode_kernel_size / 2
        kernel = kernel[None, None].float()
        inv_mask = torch.nn.functional.conv2d(inv_mask[None, None].float(), kernel, padding="same").bool()[0, 0]
        mask = torch.logical_and(mask, ~inv_mask)

    print("Making refmap from object image...", flush=True)
    refmap_est, refmask = refmap_mask_make(
        input_img[mask],
        input_normal[mask],
        res=refmap_res,
        angle_threshold=np.pi / 128 / 2,
    )

    torch.cuda.synchronize()
    print("Inpainting refmap ...", flush=True)

    batch = {
        "tag": [tag],
        "raw_refmap": refmap_est.permute(2, 0, 1)[None],
        "raw_refmask": refmask[None],
    }
    c, _, _ = ObsNet_model.get_cond_for_predict(batch)

    use_ddim = ObsNet_model.ddim_steps is not None
    with ObsNet_model.ema_scope("Plotting"):
        samples, _ = ObsNet_model.sample_log(
            cond=c,
            batch_size=len(batch["tag"]),
            ddim=use_ddim,
            ddim_steps=ObsNet_model.ddim_steps,
            eta=ObsNet_model.ddim_eta,
        )
    inpaint_sample: torch.Tensor = ObsNet_model.ds.rescale(ObsNet_model.decode_first_stage(samples))[0]

    torch.cuda.synchronize()
    print("Inverse Rendering ...", flush=True)

    batch = {
        "tag": [tag],
        "LrK": inpaint_sample[None],
    }
    LrK, _, illnet_c, refnet_c, _ = DRMNet_model.get_input_for_predict(batch)

    torch.cuda.synchronize()
    with DRMNet_model.ema_scope():
        samples, zK_est, _ = DRMNet_model.p_sample_loop(LrK, illnet_c, refnet_c, verbose=False)

    Lr0_sample: torch.Tensor = DRMNet_model.ds.rescale(DRMNet_model.decode_first_stage(samples))[0].clip(0)
    zK_est = zK_est[0]

    torch.cuda.synchronize()
    if DRMNet_model.refmap_input_scaler is not None:
        Lr0_sample /= DRMNet_model.normalizing_scale[0]

    return Lr0_sample, zK_est


def batch_validate(dataset_path: Path, output_dir: Path, obsnet_config: Path, drmnet_config: Path):
    """Process entire drmnet_synth_test dataset and compute metrics"""
    
    # Load models once
    print("Loading models...")
    obsnet_base_config = OmegaConf.load(obsnet_config)
    obsnet_model: ObsNetDiffusion = instantiate_from_config(obsnet_base_config.model).cuda()
    obsnet_model.ds: BaseDataset = instantiate_from_config(obsnet_base_config.data.params.predict)
    obsnet_model.eval()
    
    drmnet_base_config = OmegaConf.load(drmnet_config)
    drmnet_model: DRMNet = instantiate_from_config(drmnet_base_config.model).cuda()
    drmnet_model.ds: BaseDataset = instantiate_from_config(drmnet_base_config.data.params.predict)
    drmnet_model.eval()
    
    # Find all samples
    dataset_path = dataset_path / "drmnet_synth_test"
    sample_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    
    results = []
    output_dir.mkdir(exist_ok=True)
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        obj_name = sample_dir.name
        print(f"\n--- Processing {obj_name} ---")

        # Load inputs
        input_img_path = sample_dir / "obj.exr"
        input_normal_path = sample_dir / "normal.npy"
        gt_ref_path = sample_dir / "r0.exr"  # GT reflectance

        if not all(p.exists() for p in [input_img_path, input_normal_path, gt_ref_path]):
            print(f"Missing files in {obj_name}, skipping")
            continue
        
        try:
            # Load data
            input_img = load_exr(input_img_path, as_torch=True).cuda()
            input_normal = torch.from_numpy(np.load(input_normal_path)).cuda()
            normal_mask = torch.linalg.norm(input_normal, dim=-1) > 0.5
            mask = normal_mask
            
            # Run inference
            print(f"Running DRMNet inference for {obj_name}...")
            Lr0_pred, zK_est = estimate(drmnet_model, obsnet_model, input_img, input_normal, mask, tag=obj_name)
            
            # Save prediction
            pred_ref_path = os.path.join(output_dir, "preds", f"{obj_name}_r0.exr")
            os.makedirs(os.path.dirname(pred_ref_path), exist_ok=True)
            save_exr(pred_ref_path, Lr0_pred.cpu().numpy())
            
            # Compute metrics WITH input_img_path
            print(f"Computing metrics for {obj_name}...")
            metrics = compute_metrics(gt_ref_path, pred_ref_path, input_img_path)
            if metrics:
                metrics['obj_name'] = obj_name
                results.append(metrics)
                print(f"✓ RMSE: {metrics['rmse']:.4f}, SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}dB, Imax: {metrics['Imax']:.2f}")
            else:
                print(f"✗ Metrics failed for {obj_name}")
                
        except Exception as e:
            print(f"✗ Error processing {obj_name}: {e}")
            continue
    
    # Summary statistics
    if results:
        rmses = [r['rmse'] for r in results]
        ssims = [r['ssim'] for r in results]
        psnrs = [r['psnr'] for r in results]
        imaxs = [r['Imax'] for r in results]
        
        print("\n" + "="*80)
        print(f"FINAL RESULTS ({len(results)}/{len(sample_dirs)} samples):")
        print(f"Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"Mean SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        print(f"Mean PSNR: {np.mean(psnrs):.2f}dB ± {np.std(psnrs):.2f}dB")
        print(f"Mean Imax:  {np.mean(imaxs):.2f} ± {np.std(imaxs):.4f}")
        print("="*80)
        
        # Save detailed results CSV with Imax dependence
        df = pd.DataFrame(results)
        csv_path = output_dir / "metrics_vs_Imax.csv"
        df[['obj_name', 'Imax', 'rmse', 'ssim', 'psnr']].to_csv(csv_path, index=False)
        print(f"Metrics vs Imax table saved: {csv_path}")
        
        # Save full detailed results
        df.to_csv(output_dir / "validation_results.csv", index=False)
        print(f"Detailed results saved: {output_dir}/validation_results.csv")
        
        # PLOT Imax vs PSNR and Imax vs RMSE
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(imaxs, psnrs, alpha=0.7, s=30)
        plt.plot(np.unique(imaxs), np.poly1d(np.polyfit(imaxs, psnrs, 1))(np.unique(imaxs)), 
                'r-', linewidth=2, label=f'Linear fit: {np.polyfit(imaxs, psnrs, 1)[0]:.3f}x + {np.polyfit(imaxs, psnrs, 1)[1]:.2f}')
        plt.xlabel('Imax (max input light)')
        plt.ylabel('PSNR (dB)')
        plt.title('Imax vs PSNR Dependence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(imaxs, rmses, alpha=0.7, s=30, color='orange')
        plt.plot(np.unique(imaxs), np.poly1d(np.polyfit(imaxs, rmses, 1))(np.unique(imaxs)), 
                'r-', linewidth=2, label=f'Linear fit: {np.polyfit(imaxs, rmses, 1)[0]:.3f}x + {np.polyfit(imaxs, rmses, 1)[1]:.3f}')
        plt.xlabel('Imax (max input light)')
        plt.ylabel('RMSE')
        plt.title('Imax vs RMSE Dependence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plot_path = output_dir / "Imax_metrics_dependence.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dependence plots saved: {plot_path}")
        
        # Save summary
        summary = {
            'mean_rmse': np.mean(rmses),
            'std_rmse': np.std(rmses),
            'mean_ssim': np.mean(ssims),
            'std_ssim': np.std(ssims),
            'mean_psnr': np.mean(psnrs),
            'std_psnr': np.std(psnrs),
            'mean_Imax': np.mean(imaxs),
            'std_Imax': np.std(imaxs),
            'total_samples': len(results)
        }
        torch.save(summary, output_dir / 'summary.pt')
        print(f"Summary saved: {output_dir}/summary.pt")
    else:
        print("No valid results obtained!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRMNet Batch Validation")
    parser.add_argument("--dataset_path", type=Path, default=Path(".data"),
                       help="Path to drmnet_synth_test dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("./validation_outputs/"),
                       help="Output directory for predictions and metrics")
    parser.add_argument("--obsnet_base_path", type=Path, 
                       default=Path("./configs/obsnet/eval_obsnet.yaml"))
    parser.add_argument("--drmnet_base_path", type=Path, 
                       default=Path("./configs/drmnet/eval_drmnet.yaml"))
    args = parser.parse_args()
    
    print("DRMNet Batch Validation Starting...")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output:  {args.output_dir}")
    
    batch_validate(args.dataset_path, args.output_dir, args.obsnet_base_path, args.drmnet_base_path)
