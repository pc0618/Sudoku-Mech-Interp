import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm

class SudokuTrainer:
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = self.get_lr_scheduler()

    def get_lr_scheduler(self):
        def lr_lambda(step):
            if step < self.config.warmup_tokens:
                return float(step) / float(max(1, self.config.warmup_tokens))
            progress = float(step - self.config.warmup_tokens) / float(
                max(1, self.config.max_steps - self.config.warmup_tokens)
            )
            return max(
                self.config.end_lr_factor,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch, start_index):
        """Perform a single training step."""
        print(f"Batch shape: {batch.shape}")
        print(f"Start index shape: {start_index.shape}")
        print(f"Config seq_len: {self.config.seq_len}")
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device and handle input sequence
        batch = batch.to(self.device)
        start_index = start_index.to(self.device)
        
        # The full sequence is the input
        # When computing loss, we'll mask out predictions before start_index
        logits = self.model(batch)
        
        # Compute loss
        targets = batch.clone()  # Use the input as targets
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none'
        ).view(logits.size(0), -1)
        
        # Create prediction mask (only compute loss for positions after start_index)
        pred_positions = torch.arange(loss.size(1), device=self.device)[None, :]
        mask = pred_positions >= (3 * start_index)[:, None]
        
        # Apply mask and compute mean loss
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }