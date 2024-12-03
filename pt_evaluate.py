import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Literal
from torch.utils.data import Dataset, DataLoader
import os
from pt_model import *

@dataclass
class DataConfig:
    """Configuration for the Sudoku dataset."""
    seq_len: int = 243  # 3 * block_size (81)
    block_size: int = 81
    minibatch_size: int = 12
    seq_order: str = "solver-order"
    eval_epochs: int = 5
    train_puzzle_path = "datasets/train_sudoku_puzzles_full.npy"
    train_candidate_path = "datasets/train_sudoku_puzzles_candidate.npy"
    test_puzzle_path = "datasets/test_sudoku_puzzles_full.npy"
    test_candidate_path = "datasets/test_sudoku_puzzles_candidate.npy"

class SudokuDataset(Dataset):
    """PyTorch Dataset for Sudoku puzzles."""
    def __init__(self, config: DataConfig, train: bool = False):
        self.config = config
        self.train = train
        self.preprocess_sudoku()

    @classmethod
    def create_dataloader(cls, config: DataConfig, batch_size: int, train: bool = True):
        """Create a PyTorch DataLoader for the Sudoku dataset.

        Args:
            config: DataConfig object containing dataset parameters
            batch_size: Size of each batch
            train: Whether to create train or evaluation dataset

        Returns:
            DataLoader: PyTorch DataLoader object
        """
        dataset = cls(config, train=train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,  # Shuffle only training data
            num_workers=4,
            pin_memory=True
        )

    def convert_to_fixed_or_random_order(self, inputs, start_index):
        """Convert the sequence of moves to either a fixed or random order.

        Args:
            inputs: tensor of shape (num_puzzles, seq_len) containing the
                sequence of moves for each puzzle
            start_index: tensor of shape (num_puzzles, 1) containing the starting
                index for each puzzle

        Returns:
            transformed_input: tensor of shape (num_puzzles, seq_len) containing the
                sequence of moves for each puzzle in either a fixed or random order
        """
        transformed_input = torch.zeros_like(inputs)
        
        for i in range(len(inputs)):
            cur_seq = inputs[i]
            cur_start_index = start_index[i, 0]
            
            # Split the sequence into input and output prompts
            inp_prompt = cur_seq[:(3 * cur_start_index)].reshape(-1, 3)
            out_prompt = cur_seq[(3 * cur_start_index):].reshape(-1, 3)
            
            # Convert to numpy for sorting operations
            inp_prompt = inp_prompt.numpy()
            out_prompt = out_prompt.numpy()
            
            # Sort the input prompts
            if self.config.seq_order == "fixed":
                transformed_input[i, :(3 * cur_start_index)] = torch.from_numpy(
                    inp_prompt[np.lexsort(inp_prompt[:, ::-1].T)].flatten()
                )
            elif self.config.seq_order == "random":
                transformed_input[i, :(3 * cur_start_index)] = torch.from_numpy(
                    np.random.permutation(inp_prompt).flatten()
                )
            
            # Sort the output prompts
            if self.config.seq_order == "fixed":
                transformed_input[i, (3 * cur_start_index):] = torch.from_numpy(
                    out_prompt[np.lexsort(out_prompt[:, ::-1].T)].flatten()
                )
            elif self.config.seq_order == "random":
                transformed_input[i, (3 * cur_start_index):] = torch.from_numpy(
                    np.random.permutation(out_prompt).flatten()
                )
        
        return transformed_input

    def get_puzzles_start_index(self, path):
        """Get the puzzles, start index, and inputs from a given path.

        Args:
            path: the path to the file containing the puzzles

        Returns:
            inputs: tensor of shape (num_puzzles, seq_len)
            puzzles: tensor of shape (num_puzzles, block_size)
            start_index: tensor of shape (num_puzzles, 1)
        """
        # Load data and convert to numpy first to match JAX implementation
        inputs_with_start_index = np.load(path)
        start_index = inputs_with_start_index[:, 0]  # Get the start index

        inputs = inputs_with_start_index[:, 1:]
        # Delete the column corresponding to the set of strategies exactly as in JAX version
        inputs = np.delete(inputs, np.arange(81) * 4 + 3, axis=1)
        
        # Initialize puzzles array with same dtype as JAX version
        puzzles = np.zeros((len(inputs), 81), dtype=np.int8)
        for j in range(81):
            cell_id = inputs[:, 3 * j] * 9 + inputs[:, 3 * j + 1]  # Get the cell id
            puzzles[np.arange(len(inputs)), cell_id] = inputs[:, 3 * j + 2]  # Set the puzzle
        
        # Convert to torch tensors after all numpy operations are complete
        inputs = torch.from_numpy(inputs).long()
        puzzles = torch.from_numpy(puzzles).long()
        start_index = torch.from_numpy(start_index).reshape(-1, 1).long()
        
        return inputs, puzzles, start_index
        
    def preprocess_sudoku(self):
        """Preprocess the sudoku for train and test datasets."""
        if self.train:
            self.train_inputs, self.train_puzzles, self.train_start_index = (
                self.get_puzzles_start_index(self.config.train_puzzle_path)
            )
            if self.config.seq_order in {"fixed", "random"}:
                self.train_inputs = self.convert_to_fixed_or_random_order(
                    self.train_inputs, self.train_start_index
                )
        else:
            self.eval_inputs, self.eval_puzzles, self.eval_start_index = (
                self.get_puzzles_start_index(self.config.test_puzzle_path)
            )
            print(self.eval_inputs.shape, self.eval_puzzles.shape, self.eval_start_index.shape)
            if self.config.seq_order in {"fixed", "random"}:
                self.eval_inputs = self.convert_to_fixed_or_random_order(
                    self.eval_inputs, self.eval_start_index
                )

    def __len__(self):
        if self.train:
            return len(self.train_puzzles)
        else:
            return len(self.eval_puzzles)
        
    def __getitem__(self, idx):
        """Returns the training or evaluation data at the given index.

        Returns:
            A tuple containing:
            - input sequence (seq_len,)
            - puzzle solution (block_size,)
            - start index (1,)
        """
        if self.train:
            return (
                self.train_inputs[idx].long(),
                self.train_puzzles[idx].long(),
                self.train_start_index[idx].long()
            )
        else:
            return (
                self.eval_inputs[idx].long(),
                self.eval_puzzles[idx].long(),
                self.eval_start_index[idx].long()
            )



def valid_solution(output_seq):
    """
    Checks if the puzzle is a valid solution by verifying if each row, column 
    and box has all the numbers from 1 to 9.

    Args:
        output_seq: a tensor of shape (243,) containing the sequence of output numbers

    Returns:
        bool: True if correct solution, otherwise False
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(output_seq):
        output_seq = output_seq.cpu().numpy()

    # Initialize tracking arrays
    rows = np.zeros((9, 9))
    cols = np.zeros((9, 9))
    boxes = np.zeros((9, 9))

    for j in range(81):
        # Check if indices are valid
        if int(output_seq[3 * j]) >= 9:
            return False
        if int(output_seq[3 * j + 1]) >= 9:
            return False
        if int(output_seq[3 * j + 2]) > 9:
            return False
        
        row_num = int(output_seq[3 * j])
        col_num = int(output_seq[3 * j + 1])
        
        # Mark the number in the row, column and box
        rows[row_num, int(output_seq[3 * j + 2] - 1)] += 1
        cols[col_num, int(output_seq[3 * j + 2] - 1)] += 1
        boxes[
            int(3 * (row_num // 3) + (col_num // 3)), 
            int(output_seq[3 * j + 2] - 1)
        ] += 1

    return np.all(rows) and np.all(cols) and np.all(boxes)

def verify_sudoku_board(puzzle, row_num, col_num, num):
    """
    Args:
        puzzle (torch.Tensor): The correct Sudoku puzzle
        row_num (int): The row number (0-8)
        col_num (int): The column number (0-8)
        num (int): The number predicted at the specified row and column

    Raises:
        AssertionError: If the position is invalid or number doesn't match
    """
    if row_num * 9 + col_num >= 81:
        raise AssertionError("Invalid position")
    
    if puzzle[row_num * 9 + col_num] != num:
        raise AssertionError("Number mismatch")

def evaluate_model(model, eval_dataloader, config, device):
    """
    Evaluate the model on the given dataloader.

    Args:
        model: PyTorch model
        eval_dataloader: PyTorch dataloader for evaluation
        config: Configuration object
        device: torch device

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    eval_metrics = {
        "acc": [],  # Accuracy of predicting correct cell value
        "acc_complete_puzzle": []  # Accuracy of predicting correct complete puzzle
    }

    with torch.no_grad():
        for eval_epoch in range(config.eval_epochs):
            # Get batch
            batch_tuple = next(iter(eval_dataloader))
            input_seq = batch_tuple[0].to(device)
            puzzle_sol = batch_tuple[1].to(device)
            start_index = batch_tuple[2].to(device)

            total_pred, success_pred = 0, 0

            min_start_index = int(torch.min(start_index))
            cur_input_seq = input_seq[:, :(min_start_index*3)]

            for i in range(min_start_index * 3, config.seq_len):
                # Pad sequence to full length
                padding = torch.zeros(
                    (input_seq.shape[0], config.seq_len - cur_input_seq.size(1)),
                    dtype=torch.int32,
                    device=device
                )
                concat_batch = torch.cat((cur_input_seq, padding), dim=1)

                # Get model predictions
                pred_logits = model(concat_batch)

                if i % 3 == 2:
                    # Predict cell value
                    max_number = pred_logits[:, i-1, :].argmax(dim=-1)
                    mask_arr = (i >= (3 * start_index)).squeeze()

                    next_number = torch.where(
                        mask_arr,
                        max_number,
                        input_seq[:, i]
                    )

                    cur_input_seq = torch.cat(
                        (cur_input_seq, next_number.unsqueeze(1)),
                        dim=1
                    )

                    # Verify predictions
                    for j in range(len(cur_input_seq)):
                        if not mask_arr[j]:
                            continue

                        total_pred += 1
                        try:
                            verify_sudoku_board(
                                puzzle_sol[j],
                                cur_input_seq[j][i-2].item(),
                                cur_input_seq[j][i-1].item(),
                                cur_input_seq[j][i].item()
                            )
                            success_pred += 1
                        except AssertionError:
                            pass

                else:
                    # Predict row or column number
                    max_pos = pred_logits[:, i-1, :].argmax(dim=-1)
                    mask = (i >= (3 * start_index)).squeeze()
                    next_pos = torch.where(
                        mask,
                        max_pos,
                        input_seq[:, i]
                    )
                    
                    cur_input_seq = torch.cat(
                        (cur_input_seq, next_pos.unsqueeze(1)),
                        dim=1
                    )

            # Calculate accuracy metrics
            eval_metrics["acc"].append(success_pred * 1.0 / total_pred)

            # Calculate complete puzzle accuracy
            correct_eval_sudoku_puzzle = sum(
                1 for i in range(len(cur_input_seq))
                if valid_solution(cur_input_seq[i])
            )

            eval_metrics["acc_complete_puzzle"].append(
                correct_eval_sudoku_puzzle * 1.0 / len(cur_input_seq)
            )

    return eval_metrics

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize config
    config = TorchTransformerConfig()
    
    # Initialize model (assuming you have the model class from the other file)
    model = TorchTransformerLMHeadModel(config).to(device)
    model.load_state_dict(torch.load('converted_model.pt'))

    data_config = DataConfig()
    eval_dataloader = SudokuDataset.create_dataloader(
        config=data_config,
        batch_size=12,
        train=False
    )
    
    # Run evaluation
    eval_metrics = evaluate_model(model, eval_dataloader, config, device)
    
    # Print results
    print(f"Average cell prediction accuracy: {np.mean(eval_metrics['acc']):.4f}")
    print(f"Average complete puzzle accuracy: {np.mean(eval_metrics['acc_complete_puzzle']):.4f}")

if __name__ == "__main__":
    main()