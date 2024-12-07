"""Evaluation related functions."""

from flax.training import common_utils
import jax
from jax import numpy as jnp
import numpy as np

from train import model

import pdb


def valid_solution(output_seq, debug=False):
    """
    This function checks if the puzzle is a valid solution by verifying if
    each row, column and box has all the numbers from 1 to 9.

    Args:
        output_seq: a numpy array of shape (243,) containing the sequence of output numbers
        debug: if True, prints detailed debugging information

    Returns:
        bool: True if correct solution, otherwise returns False
    """
    # rows[i, j] keeps track if ith row has received (j + 1) number
    rows = np.zeros((9, 9))
    # cols[i, j] keeps track if ith column has received (j + 1) number
    cols = np.zeros((9, 9))
    # boxes[i, j] keeps track if ith box has received (j + 1) number
    boxes = np.zeros((9, 9))

    # For visualization
    if debug:
        grid = np.zeros((9, 9), dtype=int)
        print("\nProcessing moves:")

    for j in range(81):
        row_num = int(output_seq[3 * j])
        col_num = int(output_seq[3 * j + 1])
        val = int(output_seq[3 * j + 2])

        if debug:
            print(f"Move {j}: row={row_num}, col={col_num}, val={val}")

        # Check ranges
        if row_num < 0 or row_num > 8:
            if debug:
                print(f"Invalid row index {row_num} at move {j}")
            return False
        if col_num < 0 or col_num > 8:
            if debug:
                print(f"Invalid column index {col_num} at move {j}")
            return False
        if val < 1 or val > 9:
            if debug:
                print(f"Invalid value {val} at move {j}")
            return False

        if debug:
            grid[row_num, col_num] = val
            # Check if this position was already filled
            if rows[row_num, val - 1] > 0:
                print(f"Warning: Value {val} already exists in row {row_num}")
            if cols[col_num, val - 1] > 0:
                print(f"Warning: Value {val} already exists in column {col_num}")
            box_idx = 3 * (row_num // 3) + (col_num // 3)
            if boxes[box_idx, val - 1] > 0:
                print(f"Warning: Value {val} already exists in box {box_idx}")

        # Mark the number in the row, column and box
        rows[row_num, val - 1] += 1
        cols[col_num, val - 1] += 1
        boxes[
            int(3 * (row_num // 3) + (col_num // 3)), val - 1
        ] += 1

    if debug:
        # Print final grid
        print("\nFinal Grid:")
        for i in range(9):
            if i % 3 == 0:
                print("-" * 25)
            row_str = ""
            for j in range(9):
                if j % 3 == 0:
                    row_str += "| "
                row_str += f"{grid[i,j]} "
            print(row_str + "|")
        print("-" * 25)

        # Print statistics
        print("\nRow statistics:")
        for i in range(9):
            missing = np.where(rows[i] == 0)[0]
            duplicate = np.where(rows[i] > 1)[0]
            if len(missing) > 0:
                print(f"Row {i}: Missing numbers {[m+1 for m in missing]}")
            if len(duplicate) > 0:
                print(f"Row {i}: Duplicate numbers {[d+1 for d in duplicate]} (appears {[rows[i,d] for d in duplicate]} times)")

        print("\nColumn statistics:")
        for i in range(9):
            missing = np.where(cols[i] == 0)[0]
            duplicate = np.where(cols[i] > 1)[0]
            if len(missing) > 0:
                print(f"Column {i}: Missing numbers {[m+1 for m in missing]}")
            if len(duplicate) > 0:
                print(f"Column {i}: Duplicate numbers {[d+1 for d in duplicate]} (appears {[cols[i,d] for d in duplicate]} times)")

        print("\nBox statistics:")
        for i in range(9):
            missing = np.where(boxes[i] == 0)[0]
            duplicate = np.where(boxes[i] > 1)[0]
            if len(missing) > 0:
                print(f"Box {i}: Missing numbers {[m+1 for m in missing]}")
            if len(duplicate) > 0:
                print(f"Box {i}: Duplicate numbers {[d+1 for d in duplicate]} (appears {[boxes[i,d] for d in duplicate]} times)")

    valid = np.all(rows == 1) and np.all(cols == 1) and np.all(boxes == 1)
    if debug:
        print(f"\nSolution is {'valid' if valid else 'invalid'}")
        if not valid:
            if not np.all(rows == 1):
                print("Row constraint failed")
            if not np.all(cols == 1):
                print("Column constraint failed")
            if not np.all(boxes == 1):
                print("Box constraint failed")

    return valid

'''
def valid_solution(output_seq):
    """
    This function checks if the puzzle is a valid solution by verifying if
    each row, column and box has all the numbers from 1 to 9.

    Args:
        output_seq: a numpy array of shape (243,) containing the sequence of
            output numbers

    Returns:
        int: 1 if correct solution, otherwise returns 0
    """
    # rows[i, j] keeps track if ith row has received (j + 1) number
    rows = np.zeros((9, 9))
    # cols[i, j] keeps track if ith column has received (j + 1) number
    cols = np.zeros((9, 9))
    # boxes[i, j] keeps track if ith box has received (j + 1) number
    boxes = np.zeros((9, 9))

    for j in range(81):
        # The row and column are in the range (0, 8) and puzzle values are in (1, 9)
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
            int(3 * (row_num // 3) + (col_num // 3)), int(output_seq[3 * j + 2] - 1)
        ] += 1

    if np.all(rows) and np.all(cols) and np.all(boxes):
        return True
    else:
        return False
'''

def eval_step(state, batch, config):
    pred_logits = model.TransformerLMHeadModel(config).apply(
        {"params": state.params}, batch
        )
    return pred_logits


def verify_sudoku_board(puzzle, row_num, col_num, num):
    """
    Args:
        puzzle (np.array): The correct Sudoku puzzle.
        row_num (int): The row number (0-8).
        col_num (int): The column number (0-8).
        num (int): The number predicted at the specified row and column.

    Raises:
        AssertionError: If the row_num * 9 + col_num >= 81 or if the number at the specified row and column is not equal to the given number.
    """
    if row_num * 9 + col_num >= 81: 
        assert False
  
    assert puzzle[row_num * 9 + col_num] == num


def get_eval_metrics(state, eval_data_iter, p_eval_step, config):
    """This function computes given evaluation metrics (e.g, accuracy) in eval metrics for each batch and appends the metric in the list of eval_metrics.

    Args: 
        state: contains model parameters, optimizer, etc.
        eval_data_iter: data iterator for evaluation dataset
        p_eval_step: pmap function for forward pass of model for evaluation
        config: general experiment config file

    Returns: 
        eval_metrics: contains list of evaluation metrics for each batch
    """

    eval_metrics = {
        "acc": [],  # Accuracy of predicting correct cell value
        "acc_complete_puzzle": []  # Accuracy of predicting correct complete puzzle
    }

    for eval_epoch in range(config.eval_epochs):
        with jax.profiler.StepTraceAnnotation("eval", step_num=eval_epoch):

            batch_tuple = next(eval_data_iter)

            # Input seq is of the shape (batchsize, 3*81) and 3*81 because row, column
            # and value for each cell. Row, column and value all are in {1, ..., 9}
            input_seq = np.array(batch_tuple[0])

            # Puzzle solution is of the shape (batchsize, 81). Each pos in {0,.., 80}
            # for each puzzle contains value at cell (pos//9+1, pos%9 + 1)
            puzzle_sol = np.array(batch_tuple[1])
            start_index = np.array(batch_tuple[2])
            total_pred, sucess_pred = 0, 0


            min_start_index = int(np.min(start_index))
            cur_input_seq = input_seq[:, :(min_start_index*3)]
            for i in range(min_start_index * 3, config.seq_len):
                ### In i^th iteration, i^th number in sequence will predict
                padding = np.zeros((input_seq.shape[0],
                                    config.seq_len - len(cur_input_seq[0])),
                                dtype=np.int32)
                concat_batch = np.hstack((cur_input_seq, padding))
                concat_batch = common_utils.shard(
                    jax.tree_util.tree_map(np.asarray, concat_batch)
                )

                pred_logits = p_eval_step(state, concat_batch)

                if i%3 == 2:
                    # Model predicts the value at the cell (cur_input_seq[j][i-2],
                    # cur_input_seq[j][i-1])
                    max_number = pred_logits[:, :, i-1, :].argmax(axis=-1).flatten()
                    mask_arr = np.array(i >= (3 * start_index)).squeeze()

                    next_number = max_number * mask_arr + (1 - mask_arr) * input_seq[:, i]

                    cur_input_seq = np.hstack(
                        (cur_input_seq, jnp.reshape(next_number, newshape=(-1, 1)))
                    )

                    # Iterate through all examples in batch and calculate successful
                    # predictions of numbers
                    for j in range(len(cur_input_seq)):
                        if not mask_arr[j]:
                            continue

                        total_pred += 1
                        try:
                            verify_sudoku_board(puzzle_sol[j], cur_input_seq[j][i-2], 
                                                cur_input_seq[j][i-1], cur_input_seq[j][i])
                        except AssertionError:
                            # Mistake
                            pass
                        else:
                            sucess_pred += 1
                else:
                    # Model predicts either a row number or column number
                    max_pos = pred_logits[:, :, i-1, :].argmax(axis=-1).flatten()
                    mask = (i >= (3 * start_index)).squeeze()
                    next_pos = max_pos * mask + (1 - mask) * input_seq[:, i]
                    
                    # pdb.set_trace()
                    cur_input_seq = np.hstack(
                        (cur_input_seq, jnp.reshape(next_pos, newshape=(-1, 1)))
                    )

            eval_metrics["acc"].append(sucess_pred * 1.0/ total_pred)

            correct_eval_sudoku_puzzle = 0

            for i in range(len(cur_input_seq)):

                # increase correct_eval_sudoku_puzzle when the model output solution
                # for a given puzzle is correct
                correct_eval_sudoku_puzzle += valid_solution(cur_input_seq[i])

            eval_metrics["acc_complete_puzzle"].append(
                correct_eval_sudoku_puzzle * 1.0 / len(cur_input_seq)
            )

    return eval_metrics
