import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process Sudoku probe model weights.")
parser.add_argument("model_id", type=int, help="The model ID to process (e.g., 1 for model_1).")
args = parser.parse_args()

# Use the provided model ID
model_id = args.model_id

# Load your weights data
data = np.load(f'sudoku_probe_model_{model_id}_weights.npy')
print("Data shape: ", data.shape)
data = data.T

# Slice the weights
row_weights = data[0:576][:, 1:5]
col_weights = data[576:1152][:, 1:5]
val_weights = data[1152:][:, 1:5]

# Debugging: Print shapes to identify the issue
print("Shape of row_weights:", row_weights.shape)
print("Shape of col_weights:", col_weights.shape)
print("Shape of val_weights:", val_weights.shape)

# Ensure dimensions are consistent before concatenating
if row_weights.shape[1] == 4 and col_weights.shape[1] == 4 and val_weights.shape[1] == 4:
    all_weights = np.concatenate([
        row_weights.T,
        col_weights.T,
        val_weights.T
    ], axis=0)
    # Save the output file with model ID in the name
    output_file = f'all_weights_{model_id}.npy'
    np.save(output_file, all_weights)
    print(f"Saved processed weights to {output_file}")
else:
    print("Error: Inconsistent dimensions in weights. Check slicing or input data.")

