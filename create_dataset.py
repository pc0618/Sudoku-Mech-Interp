from typing import List, Set, Tuple, Optional, Dict
import argparse
import itertools
import random
import pandas as pd
from sudoku_solver import SudokuSolver
import numpy as np
def create_dataset(csv_path: str, output_path: str, max_puzzles: Optional[int] = None) -> None:
    """
    Create a numpy dataset from Sudoku puzzles.

    Args:
        csv_path: Path to the CSV file containing puzzles
        output_path: Path where to save the .npy file
        max_puzzles: Maximum number of puzzles to process (None for all)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    if max_puzzles:
        df = df.head(max_puzzles)

    dataset = []
    total = len(df)
    solved_count = 0
    unsolved_count = 0

    print(f"Processing {total} puzzles...")

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing puzzle {idx}/{total}")
            print(f"Solved so far: {solved_count}")
            print(f"Unsolved so far: {unsolved_count}")

        puzzle = row['puzzle']

        # Create solver instance
        solver = SudokuSolver(puzzle)

        # Get initial filled positions
        initial_filled = []
        for i, char in enumerate(puzzle):
            if char != '.':
                r, c = divmod(i, 9)
                initial_filled.append((r, c, int(char), solver.strategy_map["Initial"]))

        # Sort initial filled positions by row-major order
        initial_filled.sort(key=lambda x: x[0] * 9 + x[1])

        # Try to solve
        if solver.solve():
            solved_count += 1

            # Create entry for dataset
            entry = np.zeros(1 + 81 * 4, dtype=np.int32)

            # Set start index (number of initially filled cells)
            entry[0] = len(initial_filled)

            # Fill initial positions
            for i, (r, c, v, s) in enumerate(initial_filled):
                base_idx = 1 + i * 4
                entry[base_idx:base_idx + 4] = [r, c, v, s]

            # Fill solved positions
            current_idx = 1 + len(initial_filled) * 4
            for r, c, v, s in solver.moves:
                entry[current_idx:current_idx + 4] = [r, c, v, s]
                current_idx += 4

            dataset.append(entry)
        else:
            unsolved_count += 1

    if not dataset:
        raise ValueError("No puzzles were successfully solved!")

    # Convert to numpy array and save
    dataset_array = np.array(dataset)
    np.save(output_path, dataset_array)
    print(f"Successfully processed {len(dataset)} puzzles")
    print(f"Total solved: {solved_count}")
    print(f"Total unsolved: {unsolved_count}")
    print(f"Dataset shape: {dataset_array.shape}")


if __name__ == "__main__":
    # Example usage
    csv_path = "../sudoku-3m.csv"  # Path to your CSV file
    output_path = "sudoku_dataset.npy"  # Where to save the .npy file
    create_dataset(csv_path, output_path, max_puzzles=1000)  # Process first 10000 puzzles