from sudoku_solver import SudokuSolver
import csv
import random

successes = 0
total = 0
cutoff = 10000
strategy_counts = {}
with open('sudoku-3m.csv', 'r') as file:
    lines = file.readlines()
    for _ in range(cutoff):
        start, solved = lines[random.randint(0, len(lines))].split(',')[1:3]
        if start == 'puzzle':
            continue

        solver = SudokuSolver(start, None, True)
        if solver.solve():
            successes += 1
            counts = solver.get_strategy_counts()
            for key, value in counts.items():
                if key not in strategy_counts:
                    strategy_counts[key] = 0
                strategy_counts[key] += value
        total += 1

        if total % 1000 == 0:
            print(total, "puzzles solved")

print(successes, "/", total)
print("Successes rate of", successes/total)
print(strategy_counts)

