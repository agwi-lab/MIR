import torch
import torch.nn as nn
import torch.nn.functional as F


class RPredictorCNN(nn.Module):
    """
    CNN model for predicting PSNR from input images.
    Adapt architecture based on your input image dimensions.
    """
    def __init__(self, input_channels=3, initial_height=512, initial_width=512):
        super(RPredictorCNN, self).__init__()
        
        # Calculate reduced dimensions after convolutions
        def conv_output_size(size, kernel_size=3, stride=1, padding=0):
            return (size + 2*padding - kernel_size) // stride + 1
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate size after convolutions
        h = conv_output_size(initial_height, kernel_size=3, stride=2, padding=1)
        h = conv_output_size(h, kernel_size=3, stride=2, padding=1)
        h = conv_output_size(h, kernel_size=3, stride=2, padding=1)
        
        w = conv_output_size(initial_width, kernel_size=3, stride=2, padding=1)
        w = conv_output_size(w, kernel_size=3, stride=2, padding=1)
        w = conv_output_size(w, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.block = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        self.act = nn.ReLU()
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Middle part layers
        x = self.block(x)

        # Decoding (2D projection) part
        x = self.decoder(x)

        return x.squeeze()
