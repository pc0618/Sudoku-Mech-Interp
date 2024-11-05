import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

class SudokuDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.preprocess_sudoku()
    
    def convert_to_fixed_or_random_order(self, inputs, start_index):
        """Convert the sequence of moves to either a fixed or random order."""
        transformed_input = np.zeros_like(inputs)
        
        for i in range(len(inputs)):
            cur_seq = inputs[i]
            cur_start_index = start_index[i, 0]
            
            # Split the sequence into input and output prompts
            inp_prompt = cur_seq[:(3 * cur_start_index)].reshape(-1, 3)
            out_prompt = cur_seq[(3 * cur_start_index):].reshape(-1, 3)
            
            # Sort or randomize input prompts
            if self.config.seq_order == "fixed":
                transformed_input[i, :(3 * cur_start_index)] = inp_prompt[
                    np.lexsort(inp_prompt[:, ::-1].T)
                ].flatten()
            elif self.config.seq_order == "random":
                transformed_input[i, :(3 * cur_start_index)] = np.random.permutation(
                    inp_prompt
                ).flatten()
            
            # Sort or randomize output prompts
            if self.config.seq_order == "fixed":
                transformed_input[i, (3 * cur_start_index):] = out_prompt[
                    np.lexsort(out_prompt[:, ::-1].T)
                ].flatten()
            elif self.config.seq_order == "random":
                transformed_input[i, (3 * cur_start_index):] = np.random.permutation(
                    out_prompt
                ).flatten()
        
        return transformed_input

    def get_puzzles_start_index(self, path):
        """Load and preprocess puzzle data."""
        with open(path, 'rb') as f:
            inputs_with_start_index = np.load(f)
        
        # Extract start index and inputs
        start_index = inputs_with_start_index[:, 0]
        inputs = inputs_with_start_index[:, 1:]
        
        # Delete the strategy column
        inputs = np.delete(inputs, np.arange(81) * 4 + 3, axis=1)
        
        # Convert inputs to puzzle format
        puzzles = np.zeros((len(inputs), 81), dtype=np.int8)
        for j in range(81):
            cell_id = inputs[:, 3 * j] * 9 + inputs[:, 3 * j + 1]
            puzzles[np.arange(len(inputs)), cell_id] = inputs[:, 3 * j + 2]
        
        return inputs, puzzles, start_index.reshape(-1, 1)

    def preprocess_sudoku(self):
        """Preprocess the Sudoku dataset."""
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
            if self.config.seq_order in {"fixed", "random"}:
                self.eval_inputs = self.convert_to_fixed_or_random_order(
                    self.eval_inputs, self.eval_start_index
                )

    def __len__(self):
        if self.train:
            return len(self.train_puzzles)
        return len(self.eval_puzzles)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        if self.train:
            return (
                torch.tensor(self.train_inputs[idx], dtype=torch.long),
                torch.tensor(self.train_puzzles[idx], dtype=torch.long),
                torch.tensor(self.train_start_index[idx], dtype=torch.long)
            )
        return (
            torch.tensor(self.eval_inputs[idx], dtype=torch.long),
            torch.tensor(self.eval_puzzles[idx], dtype=torch.long),
            torch.tensor(self.eval_start_index[idx], dtype=torch.long)
        )

def create_dataloader(config, batch_size, train=True):
    """Create a DataLoader for the Sudoku dataset."""
    dataset = SudokuDataset(config, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Add this to ensure consistent batch sizes
    )