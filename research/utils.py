import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN model for PSNR prediction')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, 
                       default='/home/buka2004/DRMNet/validation_outputs/validation_results_cleaned.csv',
                       help='Path to CSV file with dataset information')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    # Directory arguments
    parser.add_argument('--log_dir', type=str, default='storage/results',
                        help='Directory to save experiment logs')

    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda/cpu)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()
