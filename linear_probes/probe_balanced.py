import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading data files...")
# Load data
features_data = np.load('bilinear_features_block_7_combined.npz')
moves_data = np.load('sudoku_moves.npy')
strategies_data = np.load('sudoku_strategies.npy')

# Only take first 2000 puzzles
n_puzzles = 8000
features_data = {
    'features': features_data['features'][:n_puzzles],
    'predictions': features_data['predictions'][:n_puzzles]
}
moves_data = moves_data[:n_puzzles]
strategies_data = strategies_data[:n_puzzles]

print(f"Loaded data for {n_puzzles} puzzles")

def process_puzzle_data(puzzle_idx):
    """Process data for a single puzzle, returning (features, strategies) pairs."""
    # Get features and predictions for this puzzle
    puzzle_features = features_data['features'][puzzle_idx]
    puzzle_predictions = features_data['predictions'][puzzle_idx]
    
    # Get moves and strategies for this puzzle
    puzzle_moves = moves_data[puzzle_idx].reshape(-1, 3)  # Reshape to (N, 3) for row,col,val
    puzzle_strategies = strategies_data[puzzle_idx]
    
    # Find where predictions end (first all-zero feature vector)
    non_zero_mask = np.any(puzzle_features != 0, axis=1)
    valid_count = np.sum(non_zero_mask)
    
    # Get valid predictions and features
    valid_predictions = puzzle_predictions[:valid_count].reshape(-1, 3)
    valid_features = puzzle_features[:valid_count]
    
    # Find non-initial moves (where strategy isn't [1,0,0,...])
    non_initial_mask = ~np.all(puzzle_strategies == [1,0,0,0,0,0,0,0,0,0], axis=1)
    non_initial_moves = puzzle_moves[non_initial_mask]
    non_initial_strategies = puzzle_strategies[non_initial_mask]
    
    # Create dictionary for fast lookup of strategies by move
    move_to_strategy = {tuple(move): strategy 
                       for move, strategy in zip(non_initial_moves, non_initial_strategies)}
    
    # Match features with strategies
    X, y = [], []
    for i, pred in enumerate(valid_predictions):
        move_key = tuple(pred)
        if move_key in move_to_strategy:
            # Concatenate the three consecutive feature vectors
            feature_vec = np.concatenate(valid_features[i*3:(i*3)+3])
            X.append(feature_vec)
            y.append(move_to_strategy[move_key])
    
    return np.array(X), np.array(y)

print("Processing puzzles...")
# Process all puzzles with progress bar
all_features = []
all_strategies = []
for i in tqdm(range(len(features_data['features'])), desc="Processing puzzles"):
    X, y = process_puzzle_data(i)
    all_features.append(X)
    all_strategies.append(y)

print("Concatenating data...")
# Concatenate all data
X = np.concatenate(all_features)
y = np.concatenate(all_strategies)
print(f"Total number of examples: {len(X)}")

print("Splitting into train/test sets...")
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training examples: {len(X_train)}, Test examples: {len(X_test)}")

# Create PyTorch dataset and loader
class SudokuDataset(Dataset):
    def __init__(self, features, strategies):
        self.features = torch.FloatTensor(features)
        self.strategies = torch.FloatTensor(strategies)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.strategies[idx]

# Create model
class LinearProbe(nn.Module):
    def __init__(self, input_size=1728, output_size=10):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Training parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

print("Creating datasets and loaders...")
# Create datasets and loaders
train_dataset = SudokuDataset(X_train, y_train)
test_dataset = SudokuDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("Initializing model...")
# Initialize model, loss, and optimizer
model = LinearProbe().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_start = time.time()
        
        # Add progress bar for batches within epoch
        for features, strategies in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move batch to GPU
            features = features.to(device)
            strategies = strategies.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, strategies)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f}s')
    
    # Save the model after training
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': total_loss/len(train_loader),
        'input_size': 1728,
        'output_size': 10,
    }, 'sudoku_probe_model.pt')

# Modified evaluation function with class-balanced accuracy
def evaluate():
    model.eval()
    total_correct = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for features, strategies in tqdm(test_loader, desc="Testing"):
            # Move batch to GPU
            features = features.to(device)
            strategies = strategies.to(device)
            
            outputs = model(features)
            predictions = (outputs > 0.5).float()
            
            # Move back to CPU for numpy operations
            predictions = predictions.cpu()
            strategies = strategies.cpu()
            
            all_predictions.append(predictions)
            all_targets.append(strategies)
            
            correct = (predictions == strategies).all(dim=1).sum().item()
            total_correct += correct
            total_predictions += len(features)
    
    # Calculate standard accuracy
    accuracy = total_correct / total_predictions
    print(f'Test Accuracy (all strategies correct): {accuracy:.4f}')
    
    # Calculate class-balanced accuracy
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(all_targets.shape[1]):  # For each strategy
        # Get positive and negative samples for this strategy
        positives = (all_targets[:, i] == 1)
        negatives = (all_targets[:, i] == 0)
        
        # Calculate true positives and true negatives
        tp = (all_predictions[positives, i] == 1).float().mean() if positives.any() else 0
        tn = (all_predictions[negatives, i] == 0).float().mean() if negatives.any() else 0
        
        # Balanced accuracy for this class is average of true positive and true negative rates
        balanced_acc = (tp + tn) / 2
        class_accuracies.append(balanced_acc)
    
    # Calculate mean balanced accuracy across all strategies
    mean_balanced_accuracy = sum(class_accuracies) / len(class_accuracies)
    print(f'\nClass-Balanced Test Accuracy: {mean_balanced_accuracy:.4f}')
    
    print("\nPer-strategy balanced accuracies:")
    for i, acc in enumerate(class_accuracies):
        print(f'Strategy {i}: {acc:.4f}')

def load_model(model_path='sudoku_probe_model.pt'):
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Create model with same architecture
    model = LinearProbe(input_size=checkpoint['input_size'], 
                       output_size=checkpoint['output_size'])
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

print("\nStarting training...")
train()
print("\nEvaluating model...")
evaluate()
