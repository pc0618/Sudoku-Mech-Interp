from typing import List, Set, Tuple, Optional, Dict
import argparse
import itertools
import random

class SudokuSolver:
    def __init__(self, puzzle: str, log_file: Optional[str] = None, random_order: bool = False):
        if len(puzzle) != 81:
            raise ValueError("Puzzle must be an 81-character string.")
        allowed_chars = set('1234567890.')
        if not set(puzzle).issubset(allowed_chars):
            raise ValueError("Puzzle string can only contain digits '1'-'9', '0', and '.' for blanks.")
        
        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.candidates = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
        for i, char in enumerate(puzzle):
            row, col = divmod(i, 9)
            if char in '0.':
                self.grid[row][col] = 0
                # Candidates remain as full set initially
            else:
                num = int(char)
                self.grid[row][col] = num
                self.candidates[row][col] = set()
        self.initialize_candidates()
        self.log_file = log_file
        if self.log_file:
            self.initialize_log()
        
        # Initialize strategy counts
        self.strategy_counts: Dict[str, int] = {
            "Naked Single": 0,
            "Hidden Single (Row)": 0,
            "Hidden Single (Column)": 0,
            "Hidden Single (Block)": 0,
            "Naked Pair": 0,
            "Hidden Pair": 0,
            "Pointing Pair": 0,
            "X-Wing": 0,
            "Swordfish": 0,
            # Add more strategies here as needed
        }
        
        # Set strategy order
        self.random_order = random_order
        self.strategies = [
            self.naked_singles,
            self.hidden_singles,
            self.naked_pairs,
            self.hidden_pairs,
            self.pointing_pairs,
            self.x_wing,
            self.swordfish,
            # Add more strategies here as needed
        ]
    
    def initialize_log(self):
        with open(self.log_file, 'w') as f:
            f.write("Sudoku Solver Intermediate Steps\n")
            f.write("===============================\n\n")
            f.write("Initial Puzzle:\n")
            self.log_grid(f)
            f.write("\nSolving Steps:\n\n")
    
    def log(self, message: str):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")
    
    def log_grid(self, f):
        for i, row in enumerate(self.grid):
            if i % 3 == 0 and i != 0:
                f.write("-" * 21 + "\n")
            row_display = ""
            for j, num in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_display += "| "
                row_display += f"{num if num !=0 else '.'} "
            f.write(row_display.strip() + "\n")
        f.write("\n")
    
    def initialize_candidates(self):
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] != 0:
                    self.update_candidates(row, col, self.grid[row][col])
    
    def update_candidates(self, row: int, col: int, num: int):
        # Remove the number from candidates in the same row, column, and block
        for k in range(9):
            if num in self.candidates[row][k]:
                self.candidates[row][k].discard(num)
            if num in self.candidates[k][col]:
                self.candidates[k][col].discard(num)
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if num in self.candidates[r][c]:
                    self.candidates[r][c].discard(num)
    
    def solve(self) -> bool:
        progress = True
        while progress:
            progress = False
            # Determine strategy order
            if self.random_order:
                current_strategies = self.strategies.copy()
                random.shuffle(current_strategies)
            else:
                current_strategies = self.strategies
            # Apply strategies in determined order
            for strategy in current_strategies:
                if strategy():
                    progress = True
                    break  # Restart the loop after a successful strategy application
        return self.is_solved()
    
    def naked_singles(self) -> bool:
        found = False
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0 and len(self.candidates[row][col]) == 1:
                    num = self.candidates[row][col].pop()
                    self.grid[row][col] = num
                    self.update_candidates(row, col, num)
                    self.log_step("Naked Single", num, row, col)
                    self.strategy_counts["Naked Single"] += 1
                    found = True
        return found
    
    def hidden_singles(self) -> bool:
        found = False
        # Check rows
        for row in range(9):
            counts = {}
            for col in range(9):
                for num in self.candidates[row][col]:
                    counts.setdefault(num, []).append((row, col))
            for num, positions in counts.items():
                if len(positions) == 1:
                    r, c = positions[0]
                    if self.grid[r][c] == 0:
                        self.grid[r][c] = num
                        self.candidates[r][c] = set()
                        self.update_candidates(r, c, num)
                        self.log_step("Hidden Single (Row)", num, r, c)
                        self.strategy_counts["Hidden Single (Row)"] += 1
                        found = True
        # Check columns
        for col in range(9):
            counts = {}
            for row in range(9):
                for num in self.candidates[row][col]:
                    counts.setdefault(num, []).append((row, col))
            for num, positions in counts.items():
                if len(positions) == 1:
                    r, c = positions[0]
                    if self.grid[r][c] == 0:
                        self.grid[r][c] = num
                        self.candidates[r][c] = set()
                        self.update_candidates(r, c, num)
                        self.log_step("Hidden Single (Column)", num, r, c)
                        self.strategy_counts["Hidden Single (Column)"] += 1
                        found = True
        # Check blocks
        for block in range(9):
            counts = {}
            start_row, start_col = 3 * (block // 3), 3 * (block % 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    for num in self.candidates[r][c]:
                        counts.setdefault(num, []).append((r, c))
            for num, positions in counts.items():
                if len(positions) == 1:
                    r, c = positions[0]
                    if self.grid[r][c] == 0:
                        self.grid[r][c] = num
                        self.candidates[r][c] = set()
                        self.update_candidates(r, c, num)
                        self.log_step("Hidden Single (Block)", num, r, c)
                        self.strategy_counts["Hidden Single (Block)"] += 1
                        found = True
        return found
    
    def naked_pairs(self) -> bool:
        found = False
        # Check rows
        for row in range(9):
            pairs = self.find_naked_pairs(self.candidates[row])
            if pairs:
                for num1, num2, cols in pairs:
                    for col in range(9):
                        if col not in cols and self.grid[row][col] == 0:
                            if self.candidates[row][col].intersection({num1, num2}):
                                self.candidates[row][col].discard(num1)
                                self.candidates[row][col].discard(num2)
                                self.log(f"Naked Pair in Row {row+1}: Removed {num1} and {num2} from Cell ({row+1}, {col+1})")
                                self.strategy_counts["Naked Pair"] += 1
                                found = True
        # Check columns
        for col in range(9):
            column = [self.candidates[row][col] for row in range(9)]
            pairs = self.find_naked_pairs(column)
            if pairs:
                for num1, num2, rows in pairs:
                    for row in range(9):
                        if row not in rows and self.grid[row][col] == 0:
                            if self.candidates[row][col].intersection({num1, num2}):
                                self.candidates[row][col].discard(num1)
                                self.candidates[row][col].discard(num2)
                                self.log(f"Naked Pair in Column {col+1}: Removed {num1} and {num2} from Cell ({row+1}, {col+1})")
                                self.strategy_counts["Naked Pair"] += 1
                                found = True
        # Check blocks
        for block in range(9):
            block_cells = []
            positions = []
            for r in range(3):
                for c in range(3):
                    row, col = 3 * (block // 3) + r, 3 * (block % 3) + c
                    block_cells.append(self.candidates[row][col])
                    positions.append((row, col))
            pairs = self.find_naked_pairs(block_cells)
            if pairs:
                for num1, num2, indices in pairs:
                    for idx, cell in enumerate(block_cells):
                        if idx not in indices and self.grid[positions[idx][0]][positions[idx][1]] == 0:
                            if cell.intersection({num1, num2}):
                                self.candidates[positions[idx][0]][positions[idx][1]].discard(num1)
                                self.candidates[positions[idx][0]][positions[idx][1]].discard(num2)
                                self.log(f"Naked Pair in Block {block+1}: Removed {num1} and {num2} from Cell ({positions[idx][0]+1}, {positions[idx][1]+1})")
                                self.strategy_counts["Naked Pair"] += 1
                                found = True
        return found
    
    def hidden_pairs(self) -> bool:
        found = False
        # Check rows
        for row in range(9):
            pairs = self.find_hidden_pairs(self.candidates[row])
            if pairs:
                for num1, num2, cols in pairs:
                    for col in cols:
                        before = self.candidates[row][col].copy()
                        self.candidates[row][col] = {num1, num2}
                        self.log(f"Hidden Pair in Row {row+1}: Set Cell ({row+1}, {col+1}) candidates to {num1} and {num2}")
                        if before != self.candidates[row][col]:
                            self.strategy_counts["Hidden Pair"] += 1
                            found = True
        # Check columns
        for col in range(9):
            column = [self.candidates[row][col] for row in range(9)]
            pairs = self.find_hidden_pairs(column)
            if pairs:
                for num1, num2, rows in pairs:
                    for row in rows:
                        before = self.candidates[row][col].copy()
                        self.candidates[row][col] = {num1, num2}
                        self.log(f"Hidden Pair in Column {col+1}: Set Cell ({row+1}, {col+1}) candidates to {num1} and {num2}")
                        if before != self.candidates[row][col]:
                            self.strategy_counts["Hidden Pair"] += 1
                            found = True
        # Check blocks
        for block in range(9):
            block_cells = []
            positions = []
            for r in range(3):
                for c in range(3):
                    row, col = 3 * (block // 3) + r, 3 * (block % 3) + c
                    block_cells.append(self.candidates[row][col])
                    positions.append((row, col))
            pairs = self.find_hidden_pairs(block_cells)
            if pairs:
                for num1, num2, indices in pairs:
                    for idx in indices:
                        row, col = positions[idx]
                        before = self.candidates[row][col].copy()
                        self.candidates[row][col] = {num1, num2}
                        self.log(f"Hidden Pair in Block {block+1}: Set Cell ({row+1}, {col+1}) candidates to {num1} and {num2}")
                        if before != self.candidates[row][col]:
                            self.strategy_counts["Hidden Pair"] += 1
                            found = True
        return found
    
    def pointing_pairs(self) -> bool:
        found = False
        # For each block, check if a candidate is confined to a single row or column
        for block in range(9):
            start_row, start_col = 3 * (block // 3), 3 * (block % 3)
            candidate_positions = {}
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    for num in self.candidates[r][c]:
                        candidate_positions.setdefault(num, []).append((r, c))
            for num, positions in candidate_positions.items():
                if len(positions) > 1:
                    rows = {pos[0] for pos in positions}
                    cols = {pos[1] for pos in positions}
                    if len(rows) == 1:
                        row = rows.pop()
                        for c in range(9):
                            if c < start_col or c >= start_col + 3:
                                if num in self.candidates[row][c]:
                                    self.candidates[row][c].discard(num)
                                    self.log(f"Pointing Pair in Block {block+1}: Removed {num} from Cell ({row+1}, {c+1})")
                                    self.strategy_counts["Pointing Pair"] += 1
                                    found = True
                    if len(cols) == 1:
                        col = cols.pop()
                        for r in range(9):
                            if r < start_row or r >= start_row + 3:
                                if num in self.candidates[r][col]:
                                    self.candidates[r][col].discard(num)
                                    self.log(f"Pointing Pair in Block {block+1}: Removed {num} from Cell ({r+1}, {col+1})")
                                    self.strategy_counts["Pointing Pair"] += 1
                                    found = True
        return found
    
    def x_wing(self) -> bool:
        found = False
        for num in range(1, 10):
            # Check for X-Wing in rows
            rows_with_num = []
            for row in range(9):
                cols = [col for col in range(9) if num in self.candidates[row][col]]
                if len(cols) == 2:
                    rows_with_num.append((row, cols))
            # Find X-Wing patterns
            for (row1, cols1), (row2, cols2) in itertools.combinations(rows_with_num, 2):
                if cols1 == cols2:
                    col1, col2 = cols1
                    # Eliminate num from other rows in these columns
                    for row in range(9):
                        if row != row1 and row != row2:
                            if num in self.candidates[row][col1]:
                                self.candidates[row][col1].discard(num)
                                self.log(f"X-Wing Found for {num} between Rows {row1+1} and {row2+1}: Removed {num} from Cell ({row+1}, {col1+1})")
                                self.strategy_counts["X-Wing"] += 1
                                found = True
                            if num in self.candidates[row][col2]:
                                self.candidates[row][col2].discard(num)
                                self.log(f"X-Wing Found for {num} between Rows {row1+1} and {row2+1}: Removed {num} from Cell ({row+1}, {col2+1})")
                                self.strategy_counts["X-Wing"] += 1
                                found = True
            # Check for X-Wing in columns
            cols_with_num = []
            for col in range(9):
                rows = [row for row in range(9) if num in self.candidates[row][col]]
                if len(rows) == 2:
                    cols_with_num.append((col, rows))
            for (col1, rows1), (col2, rows2) in itertools.combinations(cols_with_num, 2):
                if rows1 == rows2:
                    row1, row2 = rows1
                    # Eliminate num from other columns in these rows
                    for col in range(9):
                        if col != col1 and col != col2:
                            if num in self.candidates[row1][col]:
                                self.candidates[row1][col].discard(num)
                                self.log(f"X-Wing Found for {num} between Columns {col1+1} and {col2+1}: Removed {num} from Cell ({row1+1}, {col+1})")
                                self.strategy_counts["X-Wing"] += 1
                                found = True
                            if num in self.candidates[row2][col]:
                                self.candidates[row2][col].discard(num)
                                self.log(f"X-Wing Found for {num} between Columns {col1+1} and {col2+1}: Removed {num} from Cell ({row2+1}, {col+1})")
                                self.strategy_counts["X-Wing"] += 1
                                found = True
        return found
    
    def swordfish(self) -> bool:
        found = False
        for num in range(1, 10):
            # Check for Swordfish in rows
            rows_with_num = []
            for row in range(9):
                cols = [col for col in range(9) if num in self.candidates[row][col]]
                if 2 <= len(cols) <= 3:
                    rows_with_num.append((row, cols))
            # Find Swordfish patterns (3 rows)
            for combination in itertools.combinations(rows_with_num, 3):
                combined_cols = set(itertools.chain.from_iterable([cols for _, cols in combination]))
                if len(combined_cols) == 3:
                    # Eliminate num from other cells in these columns
                    for row in range(9):
                        if row not in [row for row, _ in combination]:
                            for col in combined_cols:
                                if num in self.candidates[row][col]:
                                    self.candidates[row][col].discard(num)
                                    self.log(f"Swordfish Found for {num} between Rows {[r+1 for r, _ in combination]}: Removed {num} from Cell ({row+1}, {col+1})")
                                    self.strategy_counts["Swordfish"] += 1
                                    found = True
            # Check for Swordfish in columns
            cols_with_num = []
            for col in range(9):
                rows = [row for row in range(9) if num in self.candidates[row][col]]
                if 2 <= len(rows) <= 3:
                    cols_with_num.append((col, rows))
            # Find Swordfish patterns (3 columns)
            for combination in itertools.combinations(cols_with_num, 3):
                combined_rows = set(itertools.chain.from_iterable([rows for _, rows in combination]))
                if len(combined_rows) == 3:
                    # Eliminate num from other cells in these rows
                    for col in range(9):
                        if col not in [col for col, _ in combination]:
                            for row in combined_rows:
                                if num in self.candidates[row][col]:
                                    self.candidates[row][col].discard(num)
                                    self.log(f"Swordfish Found for {num} between Columns {[c+1 for c, _ in combination]}: Removed {num} from Cell ({row+1}, {col+1})")
                                    self.strategy_counts["Swordfish"] += 1
                                    found = True
        return found
    
    def find_naked_pairs(self, units: List[Set[int]]) -> List[Tuple[int, int, List[int]]]:
        pairs = []
        value_to_indices = {}
        for idx, cell in enumerate(units):
            if len(cell) == 2:
                key = tuple(sorted(cell))
                value_to_indices.setdefault(key, []).append(idx)
        for pair, indices in value_to_indices.items():
            if len(indices) == 2:
                pairs.append((pair[0], pair[1], indices))
        return pairs
    
    def find_hidden_pairs(self, units: List[Set[int]]) -> List[Tuple[int, int, List[int]]]:
        pairs = []
        num_to_indices = {}
        for idx, cell in enumerate(units):
            for num in cell:
                num_to_indices.setdefault(num, set()).add(idx)
        nums = list(num_to_indices.keys())
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if num_to_indices[nums[i]] == num_to_indices[nums[j]] and len(num_to_indices[nums[i]]) == 2:
                    pairs.append((nums[i], nums[j], list(num_to_indices[nums[i]])))
        return pairs
    
    def is_solved(self) -> bool:
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    return False
        return True
    
    def get_final_state(self) -> str:
        """
        Returns the final state of the Sudoku grid as an 81-character string.
        '0' is used to represent blank cells.
        """
        return ''.join(str(num) for row in self.grid for num in row)
    
    def get_strategy_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary with the counts of how many times each strategy was applied.
        """
        return self.strategy_counts.copy()
    
    def display(self):
        for i, row in enumerate(self.grid):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            row_display = ""
            for j, num in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_display += "| "
                row_display += f"{num if num !=0 else '.'} "
            print(row_display)
        print()
    
    def log_step(self, strategy: str, num: int, row: int, col: int):
        if self.log_file:
            self.log(f"{strategy}: Placed {num} in Cell ({row+1}, {col+1})")
            with open(self.log_file, 'a') as f:
                self.log_grid(f)

def main():
    parser = argparse.ArgumentParser(description="Sudoku Solver using Human-like Strategies")
    parser.add_argument("puzzle", help="81-character puzzle string (use '0' or '.' for empty cells)")
    parser.add_argument("--log", metavar="LOG_FILE", help="Optional log file to store intermediate steps")
    parser.add_argument("--random-order", action="store_true", help="Apply strategies in random order")
    args = parser.parse_args()

    puzzle = args.puzzle
    log_file = args.log
    random_order = args.random_order

    try:
        solver = SudokuSolver(puzzle, log_file, random_order)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    print("Initial Puzzle:")
    solver.display()

    if solver.solve():
        print("Solved Puzzle:")
        solver.display()
    else:
        print("Could not solve the puzzle with the implemented strategies.")
        print("Current state:")
        solver.display()

    # Print the final state as a string
    final_state = solver.get_final_state()
    print(f"Final State as String: {final_state}")

    # Print strategy counts
    strategy_counts = solver.get_strategy_counts()
    print("Strategy Usage Counts:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")

    if log_file:
        print(f"Intermediate steps have been logged to '{log_file}'.")

if __name__ == "__main__":
    main()

