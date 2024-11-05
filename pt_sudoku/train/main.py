import os
import argparse
import logging
import torch
import wandb
from dataclasses import dataclass
import random
import numpy as np
from typing import Optional

import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the train_and_evaluate function using the absolute import
from train.train_and_evaluate import train_and_evaluate

logging.basicConfig(level=logging.INFO)

@dataclass
class Config:
    """Configuration for the Sudoku solver experiment."""
    # Dataset choice
    dataset: str = 'sudoku'
    
    # Sequence order
    seq_order: str = "solver-order"  # Choices = ["fixed", "solver-order", "random"]
    
    # Training related parameters
    max_steps: int = 2**22
    dtype: torch.dtype = torch.float32
    minibatch_size: int = 64
    
    # Model related parameters
    block_size: int = 81
    seq_len: int = 3 * block_size
    vocab_size: int = 11
    
    # Model architecture
    num_heads: int = 8
    num_layers: int = 8
    emb_dim: int = 576
    qkv_dim: int = 576
    mlp_dim: int = 6 * emb_dim
    dropout_rate: float = 0.2
    attention_dropout_rate: float = 0.2
    
    # Training hyperparameters
    learning_rate: float = 0.0002
    end_lr_factor: float = 0.2
    warmup_tokens: int = 10000
    weight_decay: float = 0.005
    resume_training: bool = False
    
    # Other hyperparameters
    seed: int = 7
    save_checkpoint: bool = True
    save_every_steps: int = 32000
    use_wandb: bool = False
    wandb_project_name: str = 'sudoku'
    
    # Evaluation related parameters
    eval_every_steps: int = 32000
    eval_epochs: int = 5
    
    # Path to dataset
    train_puzzle_path: str = "datasets/train_sudoku_puzzles.npy"
    train_candidate_path: str = "datasets/train_sudoku_puzzles_candidate.npy"
    test_puzzle_path: str = "datasets/test_sudoku_puzzles.npy"
    test_candidate_path: str = "datasets/test_sudoku_puzzles_candidate.npy"
    
    # Additional fields for experiment management
    workdir: Optional[str] = None
    ckpt_loc: Optional[str] = None

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Sudoku GPT experiments')
    parser.add_argument('--workdir', type=str, required=True,
                      help='Directory to store model data')
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Experiment name')
    parser.add_argument('--ckpt_loc', type=str,
                      help='Directory to restore model')
    parser.add_argument('--config_path', type=str,
                      help='Path to config file (not implemented yet)')
    
    args = parser.parse_args()
    
    # Get default config
    config = Config()
    
    # Update config with command line arguments
    config.workdir = args.workdir
    config.ckpt_loc = args.ckpt_loc
    
    if config.resume_training:
        assert args.ckpt_loc is not None, "Must provide checkpoint location when resuming training"
    
    # Set up wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project_name,
            name=args.exp_name,
            config=vars(config)
        )
    
    # Create working directory if it doesn't exist
    os.makedirs(config.workdir, exist_ok=True)
    
    # Set up logging
    logging.info('PyTorch device count: %d', torch.cuda.device_count() if torch.cuda.is_available() else 0)
    logging.info('Using device: %s', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Log the config
    logging.info('Configuration:')
    for key, value in vars(config).items():
        logging.info('%s: %s', key, value)
    
    # Set random seed
    set_seed(config.seed)    
    
    try:
        # Start training
        train_and_evaluate(config, config.workdir)
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
    except Exception as e:
        logging.error('Error during training: %s', str(e))
        raise
    finally:
        if config.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()