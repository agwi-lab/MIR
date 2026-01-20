import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Assuming these are defined elsewhere

# Import from utils
from loss import RLoss
from data import create_data_loaders
from model import RPredictorCNN
from utils import parse_args


class RTrainer:
    def __init__(self, model, train_loader, val_loader=None, 
                 device='cuda', log_dir='results', args=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = RLoss()

        lr = 3e-4 if args is None else args.learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Cosine annealing scheduler - FIXED
        num_epochs = 50 if args is None else args.epochs
        # T_max is the number of epochs until restart, eta_min is minimum learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,  # Use the total number of epochs
            eta_min=1e-6,      # Minimum learning rate
            last_epoch=-1
        )

        # Directories
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.tensorboard_dir = os.path.join(log_dir, 'runs')
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.tensorboard_dir, f"experiment_{timestamp}"))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
    def save_checkpoint(self, epoch, is_best=False, additional_info=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        if additional_info:
            checkpoint.update(additional_info)

        # Save regular checkpoint
        if epoch % 100 == 0:
          checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
          torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")

        # Also save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found!")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        batch_losses = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch["input"].to(self.device)
            psnr = batch["psnr"].to(self.device)
            rmse = batch["rmse"].to(self.device)
            Imax = batch["Imax"].to(self.device)
            N = batch["N"].to(self.device)

            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, rmse, psnr, Imax, N)
            loss = loss_dict['total_loss']
            rmse_loss = loss_dict['rmse_loss']
            psnr_loss = loss_dict['psnr_loss']

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch loss to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
            self.writer.add_scalar('train/batch_rmse_loss', rmse_loss.item(), global_step)
            self.writer.add_scalar('train/batch_psnr_loss', psnr_loss.item(), global_step)
        
        epoch_loss = running_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
        
        # Log epoch metrics to TensorBoard
        self.writer.add_scalar('train/epoch_loss', epoch_loss, self.current_epoch)
        self.writer.add_scalar('train/learning_rate', 
                              self.optimizer.param_groups[0]['lr'], 
                              self.current_epoch)

        # Note: To log epoch-level RMSE and PSNR losses, we would need to accumulate them
        # throughout the epoch, but since we only have batch-level losses from RLoss,
        # we log the batch losses above and could compute epoch averages if needed

        return epoch_loss
    
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        running_loss = 0.0
        running_rmse_loss = 0.0
        running_psnr_loss = 0.0
        all_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                inputs = batch["input"].to(self.device)
                psnr = batch["psnr"].to(self.device)
                rmse = batch["rmse"].to(self.device)
                Imax = batch["Imax"].to(self.device)
                N = batch["N"].to(self.device)
                
                outputs = self.model(inputs)

                loss_dict = self.criterion(outputs, rmse, psnr, Imax, N)
                loss = loss_dict['total_loss']
                rmse_loss = loss_dict['rmse_loss']
                psnr_loss = loss_dict['psnr_loss']

                running_loss += loss.item()
                running_rmse_loss += rmse_loss.item()
                running_psnr_loss += psnr_loss.item()
                
                pbar.set_postfix({'val_loss': loss.item()})
                
                all_predictions.extend(outputs.cpu().numpy())
        
        epoch_val_loss = running_loss / len(self.val_loader)
        epoch_rmse_loss = running_rmse_loss / len(self.val_loader)
        epoch_psnr_loss = running_psnr_loss / len(self.val_loader)
        
        self.val_losses.append(epoch_val_loss)
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        
        # Log validation metrics to TensorBoard
        self.writer.add_scalar('val/epoch_loss', epoch_val_loss, self.current_epoch)
        self.writer.add_scalar('val/epoch_rmse_loss', epoch_rmse_loss, self.current_epoch)
        self.writer.add_scalar('val/epoch_psnr_loss', epoch_psnr_loss, self.current_epoch)
        
        # Log predictions vs targets histogram
        self.writer.add_histogram('val/predictions', all_predictions, self.current_epoch)
        
        return epoch_val_loss
    
    def train(self, num_epochs=50, resume_from=None):
        """Train the model"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"TensorBoard directory: {self.tensorboard_dir}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            if self.val_loader:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.6f}")
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.6f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.save_checkpoint(epoch)

            # Update learning rate with cosine annealing scheduler - FIXED
            self.scheduler.step()
            
            # Print learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Log model weights histograms to TensorBoard every 5 epochs
            if (epoch + 1) % 5 == 0:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'weights/{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
        
        # Close TensorBoard writer
        self.writer.close()
        print(f"\nTraining completed!")
        print(f"TensorBoard logs saved to: {self.writer.log_dir}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        args.csv_path, 
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed
    )

    # Get sample to determine input dimensions
    sample = train_loader.dataset[0]
    input_sample = sample["input"]
    print(f"Input shape: {input_sample.shape}")
    
    # Create model
    input_channels = input_sample.shape[0]
    height = input_sample.shape[1]
    width = input_sample.shape[2]
    
    model = RPredictorCNN(
        input_channels=input_channels,
        initial_height=height,
        initial_width=width
    )
    
    print(f"Model created with input channels: {input_channels}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = RTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=args.log_dir,
        args=args
    )

    # Set learning rate if specified
    if args.learning_rate != 0.001:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        print(f"Set learning rate to: {args.learning_rate}")
    
    # Train model
    trainer.train(num_epochs=args.epochs, resume_from=args.resume_from)

    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,
        'args': vars(args)
    }

    # Set dirs
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    tensorboard_dir = os.path.join(args.log_dir, 'runs')
    
    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved to {final_path}")

    # Print TensorBoard command
    print(f"\nTo visualize training logs, run:")
    print(f"tensorboard --logdir={tensorboard_dir}")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"- Checkpoints saved in: {checkpoint_dir}")
    print(f"- TensorBoard logs in: {tensorboard_dir}")
    print(f"- Best validation loss: {trainer.best_val_loss:.6f}")
    
    if len(trainer.train_losses) > 0:
        print(f"- Final train loss: {trainer.train_losses[-1]:.6f}")
    if len(trainer.val_losses) > 0:
        print(f"- Final validation loss: {trainer.val_losses[-1]:.6f}")


if __name__ == "__main__":
    main()