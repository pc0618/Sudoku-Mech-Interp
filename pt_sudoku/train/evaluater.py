import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from train.model import TransformerLMHeadModel

def valid_solution(output_seq):
    """Check if the Sudoku solution is valid."""
    rows = np.zeros((9, 9))
    cols = np.zeros((9, 9))
    boxes = np.zeros((9, 9))
    
    for j in range(81):
        if int(output_seq[3 * j]) >= 9:
            return False
        if int(output_seq[3 * j + 1]) >= 9:
            return False
        if int(output_seq[3 * j + 2]) > 9:
            return False
        
        row_num = int(output_seq[3 * j])
        col_num = int(output_seq[3 * j + 1])
        
        rows[row_num, int(output_seq[3 * j + 2] - 1)] += 1
        cols[col_num, int(output_seq[3 * j + 2] - 1)] += 1
        boxes[
            int(3 * (row_num // 3) + (col_num // 3)),
            int(output_seq[3 * j + 2] - 1)
        ] += 1
    
    return np.all(rows) and np.all(cols) and np.all(boxes)

def verify_sudoku_board(puzzle, row_num, col_num, num):
    """Verify if a number placement is correct according to the solution."""
    if row_num * 9 + col_num >= 81:
        return False
    return puzzle[row_num * 9 + col_num] == num

class SudokuEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_step(self, batch):
        """Perform a single evaluation step."""
        self.model.eval()
        with torch.no_grad():
            return self.model(batch.to(self.device))

    def evaluate(self, eval_loader, config):
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        eval_metrics = {
            'acc': [],
            'acc_complete_puzzle': []
        }
        
        with torch.no_grad():
            for _ in range(config.eval_epochs):
                batch, puzzle_sol, start_index = next(iter(eval_loader))
                batch = batch.to(self.device)
                puzzle_sol = puzzle_sol.to(self.device)
                start_index = start_index.to(self.device)
                
                min_start_index = int(torch.min(start_index))
                cur_input_seq = batch[:, :(min_start_index*3)]
                
                total_pred = 0
                success_pred = 0
                
                # Generate predictions
                for i in range(min_start_index * 3, config.seq_len):
                    padding = torch.zeros(
                        (batch.size(0), config.seq_len - cur_input_seq.size(1)),
                        dtype=torch.long,
                        device=self.device
                    )
                    concat_batch = torch.cat([cur_input_seq, padding], dim=1)
                    
                    pred_logits = self.evaluate_step(concat_batch)
                    
                    if i % 3 == 2:
                        max_number = pred_logits[:, i-1, :].argmax(dim=-1)
                        mask_arr = (i >= (3 * start_index)).squeeze()
                        
                        next_number = torch.where(
                            mask_arr,
                            max_number,
                            batch[:, i]
                        )
                        
                        cur_input_seq = torch.cat(
                            [cur_input_seq, next_number.unsqueeze(1)],
                            dim=1
                        )
                        
                        # Calculate accuracy
                        for j in range(len(cur_input_seq)):
                            if not mask_arr[j]:
                                continue
                            
                            total_pred += 1
                            if verify_sudoku_board(
                                puzzle_sol[j],
                                cur_input_seq[j][i-2].item(),
                                cur_input_seq[j][i-1].item(),
                                cur_input_seq[j][i].item()
                            ):
                                success_pred += 1
                    else:
                        max_pos = pred_logits[:, i-1, :].argmax(dim=-1)
                        mask = (i >= (3 * start_index)).squeeze()
                        next_pos = torch.where(mask, max_pos, batch[:, i])
                        cur_input_seq = torch.cat(
                            [cur_input_seq, next_pos.unsqueeze(1)],
                            dim=1
                        )
                
                # Calculate metrics
                eval_metrics['acc'].append(success_pred / max(1, total_pred))
                
                correct_puzzles = sum(
                    valid_solution(seq.cpu().numpy())
                    for seq in cur_input_seq
                )
                eval_metrics['acc_complete_puzzle'].append(
                    correct_puzzles / len(cur_input_seq)
                )
        
        return eval_metrics