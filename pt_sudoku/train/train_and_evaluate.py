import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import math

# Import from sibling modules in the train package
from .data import create_dataloader
from .model import TransformerLMHeadModel
from .trainer import SudokuTrainer
from .evaluater import SudokuEvaluator

def train_and_evaluate(config, workdir):
    """Main training and evaluation loop."""
    # Create data loaders
    train_loader = create_dataloader(config, config.minibatch_size, train=True)
    eval_loader = create_dataloader(config, config.minibatch_size, train=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLMHeadModel(config).to(device)
    
    # Initialize trainer and evaluator
    trainer = SudokuTrainer(model, config)
    evaluator = SudokuEvaluator(model, config)
    
    # Setup logging
    writer = SummaryWriter(workdir)
    if config.use_wandb:
        wandb.init(project="sudoku-solver", config=vars(config))
    
    # Log hyperparameters
    total_params = sum(p.numel() for p in model.parameters())
    writer.add_text('hyperparameters/model_size', f'Total parameters: {total_params}')
    
    # Training loop
    best_eval_acc = 0
    steps_since_eval = 0
    
    for step in tqdm(range(config.max_steps)):
        # Training step
        batch, _, start_index = next(iter(train_loader))
        metrics = trainer.train_step(batch, start_index)
        
        # Log metrics
        writer.add_scalar('training/loss', metrics['loss'], step)
        writer.add_scalar('training/learning_rate', metrics['learning_rate'], step)
        
        if config.use_wandb:
            wandb.log({
                'loss': metrics['loss'],
                'learning_rate': metrics['learning_rate']
            }, step=step)
        
        # Evaluation
        if step % config.eval_every_steps == 0:
            eval_metrics = evaluator.evaluate(eval_loader, config)
            
            # Log evaluation metrics
            for key, values in eval_metrics.items():
                avg_value = np.mean(values)
                writer.add_scalar(f'eval/{key}', avg_value, step)
                if config.use_wandb:
                    wandb.log({f'eval_{key}': avg_value}, step=step)
            
            # Save checkpoint if better than previous best
            avg_acc = np.mean(eval_metrics['acc'])
            if avg_acc > best_eval_acc and config.save_checkpoint:
                best_eval_acc = avg_acc
                checkpoint_path = os.path.join(workdir, f'checkpoint_step_{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'best_eval_acc': best_eval_acc,
                }, checkpoint_path)
        
        # Check for NaN loss
        if math.isnan(metrics['loss']):
            print("Loss became NaN. Training stopped.")
            break
    
    writer.close()
    if config.use_wandb:
        wandb.finish()