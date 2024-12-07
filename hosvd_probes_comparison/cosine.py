import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Calculate cosine similarities for Sudoku probes.")
parser.add_argument("model_id", type=int, help="The model ID for input files (e.g., 1 for output_directions_1.npy).")
args = parser.parse_args()

# Use the provided model ID
model_id = args.model_id

# Load files dynamically
output_directions_file = f'output_directions_{model_id}.npy'
all_weights_file = f'all_weights_{model_id}.npy'

output_directions = np.load(output_directions_file)
probes = np.load(all_weights_file)

print("Shape of output_directions:", output_directions.shape)  # Should be 5 x 576
print("Shape of probes:", probes.shape)  # Should be 12 x 576

# Process the data
output_directions = output_directions[0:5]  # Slice the first 5 directions
final = np.zeros((12, 5))

# Calculate cosine similarity
for i in range(probes.shape[0]):
    for j in range(output_directions.shape[0]):
        final[i, j] = (probes[i] @ output_directions[j]) / (
            np.linalg.norm(probes[i]) * np.linalg.norm(output_directions[j])
        )

# Display results
print("Final cosine similarity matrix:")
print(final)
print("Maximum cosine similarity:", np.max(final))

