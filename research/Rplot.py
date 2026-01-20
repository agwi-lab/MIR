import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
from scipy import stats

# Add the directory containing data.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from data import RDataset
from model import RPredictorCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Predict R values and create 3D plot')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 4)')
    parser.add_argument('--output', type=str, default='3d_plot.png',
                       help='Output filename for the plot (default: 3d_plot.png)')
    parser.add_argument('--out-dir', type=str, default='r-plot-dir',
                    help='Output results of the 3D plot')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    return parser.parse_args()

def extract_2d_features(model, dataloader, device):
    """
    Extract 2D features from the model (before the decoder).
    Returns: (features_2d, predictions_r)
    """
    model.eval()
    all_features = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Data processing'):
            inputs = batch["input"].to(device)

            # Forward pass through encoder and block (but not decoder)
            x = model.encoder(inputs)
            x = model.global_pool(x)
            x = x.view(x.size(0), -1)
            x = model.block(x)  # This gives us the 2D features

            # Get predictions (through decoder)
            predictions = model.decoder(x).squeeze()
            
            all_features.append(x.cpu().numpy())

            pred_np = predictions.cpu().numpy()
            if pred_np.ndim == 0:  # Scalar case
                pred_np = np.array([pred_np])  # Convert to 1D array
            all_predictions.append(pred_np)

    # Concatenate all batches
    features_2d = np.vstack(all_features)
    predictions_r = np.concatenate(all_predictions)
    
    return features_2d, predictions_r

def plot_3d_projection(features_2d, predictions_r, output_path='3d_plot.png', outlier_threshold=3.0):
    """
    Create a 3D plot with 2D features as x,y and predictions as z.
    Filters out outliers based on z-score threshold.
    """
    # Convert to numpy arrays if not already
    features_2d = np.array(features_2d)
    predictions_r = np.array(predictions_r)
    
    # Calculate z-scores for the predictions (z-coordinate)
    z_scores = np.abs(stats.zscore(predictions_r))
    
    # Identify non-outliers (where z-score is below threshold)
    non_outlier_mask = z_scores < outlier_threshold
    outlier_mask = z_scores >= outlier_threshold

    # Count outliers
    n_outliers = np.sum(outlier_mask)
    n_total = len(predictions_r)

    # Print info about filtered outliers
    print(f"Total points before filtering: {n_total}")
    print(f"Outliers filtered (z-score >= {outlier_threshold}): {n_outliers}")
    print(f"Percentage of outliers: {(n_outliers/n_total)*100:.2f}%")

    if n_outliers > 0:
        print(f"\nOutlier statistics:")
        print(f"  Min outlier value: {predictions_r[outlier_mask].min():.4f}")
        print(f"  Max outlier value: {predictions_r[outlier_mask].max():.4f}")
        print(f"  Mean outlier value: {predictions_r[outlier_mask].mean():.4f}")
        print(f"  Median of all data: {np.median(predictions_r):.4f}")
        print(f"  Mean of all data: {predictions_r.mean():.4f}")
        print(f"  Std of all data: {predictions_r.std():.4f}")

    # Filter data
    filtered_features = features_2d[non_outlier_mask]
    filtered_predictions = predictions_r[non_outlier_mask]
    filtered_z_scores = z_scores[non_outlier_mask]
    
    print(f"\nPoints plotted after filtering: {len(filtered_predictions)}")
    
    # Create plot with filtered data
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x and y coordinates from filtered 2D features
    x_coords = filtered_features[:, 0]
    y_coords = filtered_features[:, 1]
    z_coords = filtered_predictions
    
    # Create scatter plot with filtered data
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=z_coords, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Predicted R Value', fontsize=12)
    
    # Set labels
    ax.set_xlabel('2D Feature Dimension 1', fontsize=12, labelpad=10)
    ax.set_ylabel('2D Feature Dimension 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Predicted R', fontsize=12, labelpad=10)
    
    # Set title with outlier info
    title = f'3D Projection: 2D Features vs Predicted R\n(Filtered {n_outliers} outliers, z-score ≥ {outlier_threshold})'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = f'Plotted: {len(filtered_predictions)} points\nRemoved: {n_outliers} outliers'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D plot saved to {output_path}")
    
    # Optional: Return filtered data and outlier info for further analysis
    outlier_info = {
        'n_total': n_total,
        'n_outliers': n_outliers,
        'outlier_indices': np.where(outlier_mask)[0],
        'outlier_values': predictions_r[outlier_mask],
        'filtered_features': filtered_features,
        'filtered_predictions': filtered_predictions,
        'z_scores': z_scores
    }

    return outlier_info

def plot_3d_projection_pro(features_2d, predictions_r, output_path='3d_plot.png', outlier_threshold=3.0):
    """
    Create a 3D plot with 2D features as x,y and predictions as z.
    Filters out outliers based on z-score threshold.
    Shows multiple viewpoints in subplots.
    """
    # Convert to numpy arrays if not already
    features_2d = np.array(features_2d)
    predictions_r = np.array(predictions_r)
    
    # Calculate z-scores for the predictions (z-coordinate)
    z_scores = np.abs(stats.zscore(predictions_r))
    
    # Identify non-outliers (where z-score is below threshold)
    non_outlier_mask = z_scores < outlier_threshold
    outlier_mask = z_scores >= outlier_threshold

    # Count outliers
    n_outliers = np.sum(outlier_mask)
    n_total = len(predictions_r)

    # Print info about filtered outliers
    print(f"Total points before filtering: {n_total}")
    print(f"Outliers filtered (z-score >= {outlier_threshold}): {n_outliers}")
    print(f"Percentage of outliers: {(n_outliers/n_total)*100:.2f}%")

    if n_outliers > 0:
        print(f"\nOutlier statistics:")
        print(f"  Min outlier value: {predictions_r[outlier_mask].min():.4f}")
        print(f"  Max outlier value: {predictions_r[outlier_mask].max():.4f}")
        print(f"  Mean outlier value: {predictions_r[outlier_mask].mean():.4f}")
        print(f"  Median of all data: {np.median(predictions_r):.4f}")
        print(f"  Mean of all data: {predictions_r.mean():.4f}")
        print(f"  Std of all data: {predictions_r.std():.4f}")

    # Filter data
    filtered_features = features_2d[non_outlier_mask]
    filtered_predictions = predictions_r[non_outlier_mask]
    filtered_z_scores = z_scores[non_outlier_mask]
    
    print(f"\nPoints plotted after filtering: {len(filtered_predictions)}")
    
    # Create plot with multiple viewpoints - Increased figure size
    fig = plt.figure(figsize=(22, 14))  # Increased from (18, 10)
    
    # Define viewpoints: (elevation, azimuth)
    viewpoints = [
        (20, 45),   # Standard view
        (20, 135),  # Side view
        (20, 225),  # Back view
        (20, 315),  # Other side view
        (70, 45),   # Top-down view
        (0, 0)      # Front view
    ]
    
    titles = [
        'Standard View (45°)',
        'Side View (135°)', 
        'Back View (225°)',
        'Other Side (315°)',
        'Top-Down View',
        'Front View'
    ]
    
    # Create 6 subplots
    for i, ((elev, azim), title) in enumerate(zip(viewpoints, titles), 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        
        # Extract coordinates
        x_coords = filtered_features[:, 0]
        y_coords = filtered_features[:, 1]
        z_coords = filtered_predictions
        
        # Create scatter plot
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                            c=z_coords, cmap='viridis', 
                            s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # Set labels with increased padding
        ax.set_xlabel('Feature Dim 1', fontsize=10, labelpad=12)  # Increased labelpad
        ax.set_ylabel('Feature Dim 2', fontsize=10, labelpad=12)  # Increased labelpad
        ax.set_zlabel('Predicted R', fontsize=10, labelpad=12)    # Increased labelpad
        
        # Set title with increased padding
        ax.set_title(title, fontsize=11, pad=15)  # Increased pad
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add grid
        ax.grid(True, alpha=0.2)
    
    # Add colorbar with adjusted position
    plt.tight_layout(rect=[0, 0, 0.88, 0.92])  # Adjusted rect for more space
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # Moved further right
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Predicted R Value', fontsize=12, labelpad=10)
    
    # Add main title with adjusted position
    fig.suptitle(f'3D Projection: 2D Features vs Predicted R\n'
                f'(Filtered {n_outliers} outliers, z-score ≥ {outlier_threshold})', 
                fontsize=14, y=0.96)  # Adjusted y position
    
    # Add statistics text box with adjusted position and size
    stats_text = f'Plotted: {len(filtered_predictions)} points\nRemoved: {n_outliers} outliers\n'
    stats_text += f'R range: [{filtered_predictions.min():.2f}, {filtered_predictions.max():.2f}]'
    fig.text(0.02, 0.95, stats_text, transform=fig.transFigure,  # Adjusted y position
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D plot with multiple views saved to {output_path}")
    
    # Optional: Return filtered data and outlier info for further analysis
    outlier_info = {
        'n_total': n_total,
        'n_outliers': n_outliers,
        'outlier_indices': np.where(outlier_mask)[0],
        'outlier_values': predictions_r[outlier_mask],
        'filtered_features': filtered_features,
        'filtered_predictions': filtered_predictions,
        'z_scores': z_scores
    }

    return outlier_info

def main():
    args = parse_args()

    # out preprocess
    os.makedirs(args.out_dir, exist_ok=True)
    args.output = os.path.join(args.out_dir, args.output)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:4" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Check if files exist
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Create dataset and dataloader
    print(f"Loading dataset from: {args.dataset}")
    dataset = RDataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent plotting
        num_workers=2
    )
    
    # Initialize model
    print(f"Loading model from: {args.checkpoint}")
    model = RPredictorCNN()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)

    # Extract 2D features and predictions
    print("Extracting 2D features and making predictions...")
    features_2d, predictions_r = extract_2d_features(model, dataloader, device)

    # Print statistics
    print(f"\nExtracted {len(features_2d)} samples")
    print(f"2D feature range: [{features_2d.min():.4f}, {features_2d.max():.4f}]")
    print(f"2D feature mean: [{features_2d[:, 0].mean():.4f}, {features_2d[:, 1].mean():.4f}]")
    print(f"Predicted R range: [{predictions_r.min():.4f}, {predictions_r.max():.4f}]")
    print(f"Predicted R mean: {predictions_r.mean():.4f}")

    # Create 3D plot
    print(f"\nCreating 3D plot...")
    plot_3d_projection_pro(features_2d, predictions_r, args.output, outlier_threshold=7.0)
    
    # Optional: Save data to CSV for further analysis
    data_df = pd.DataFrame({
        'feature_dim1': features_2d[:, 0],
        'feature_dim2': features_2d[:, 1],
        'predicted_r': predictions_r
    })
    csv_path = args.output.replace('.png', '_data.csv')
    data_df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

if __name__ == "__main__":
    main()
