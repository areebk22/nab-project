import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from nab.detectors.base import AnomalyDetector

class TimeSeriesTransformerDetector(AnomalyDetector, nn.Module):
    def __init__(self, *args, **kwargs):
        super(TimeSeriesTransformerDetector, self).__init__(*args, **kwargs)
        nn.Module.__init__(self)  # Initialize nn.Module
        
        # Hyperparameters
        self.window_size = 128  # Input sequence length
        self.d_model = 32       # Embedding dimension
        self.nhead = 4          # Number of attention heads
        self.num_layers = 2     # Number of transformer layers
        
        # Model components
        self.encoder = nn.Linear(1, self.d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True  # Fixes the warning
        )
        self.transformer = TransformerEncoder(encoder_layers, self.num_layers)
        self.decoder = nn.Linear(self.d_model, 1)
        self.loss_fn = nn.MSELoss()
        
        # Data buffer
        self.buffer = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def handleRecord(self, inputData):
        # Append new data point
        self.buffer.append(inputData["value"])
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Initialize anomaly score
        anomaly_score = 0.0
        
        # Train/predict when buffer is full
        if len(self.buffer) == self.window_size:
            # Convert buffer to tensor
            sequence = torch.FloatTensor(self.buffer).unsqueeze(-1)  # (seq_len, 1)
            
            # Forward pass
            encoded = self.encoder(sequence)          # (seq_len, d_model)
            transformer_out = self.transformer(encoded)  # (seq_len, d_model)
            decoded = self.decoder(transformer_out)  # (seq_len, 1)
            
            # Compute reconstruction error
            loss = self.loss_fn(decoded, sequence)
            anomaly_score = loss.item()
            
            # Online training (update model incrementally)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Normalize score to [0, 1]
            anomaly_score = min(1.0, anomaly_score * 10)  # Adjust scaling factor
        
        return (anomaly_score, )