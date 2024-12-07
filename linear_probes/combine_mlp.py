import numpy as np

def combine_feature_files(block_num, num_files=13):
    """Combine feature files while maintaining order."""
    # First pass: get total size and verify shapes
    total_puzzles = 0
    feature_dim = None
    pred_dim = None
    
    print("Verifying file shapes...")
    for i in range(num_files):
        data = np.load(f'bilinear_features_block_{block_num}_{i}.npz')
        n_puzzles = data['features'].shape[0]
        total_puzzles += n_puzzles
        
        # Verify dimensions are consistent
        if feature_dim is None:
            feature_dim = data['features'].shape[1:]
            pred_dim = data['predictions'].shape[1:]
        else:
            assert data['features'].shape[1:] == feature_dim, f"Inconsistent feature dimensions in file {i}"
            assert data['predictions'].shape[1:] == pred_dim, f"Inconsistent prediction dimensions in file {i}"
        
        print(f"File {i}: {n_puzzles} puzzles")
    
    print(f"\nTotal puzzles to combine: {total_puzzles}")
    
    # Pre-allocate arrays
    features = np.zeros((total_puzzles, *feature_dim))
    predictions = np.zeros((total_puzzles, *pred_dim), dtype=np.int32)
    
    # Second pass: fill arrays
    current_idx = 0
    for i in range(num_files):
        data = np.load(f'bilinear_features_block_{block_num}_{i}.npz')
        n_puzzles = data['features'].shape[0]
        
        # Fill slices of the arrays
        features[current_idx:current_idx + n_puzzles] = data['features']
        predictions[current_idx:current_idx + n_puzzles] = data['predictions']
        print(f"Added file {i} at indices {current_idx} to {current_idx + n_puzzles - 1}")
        current_idx += n_puzzles
    
    print(f"\nFinal shapes:")
    print(f"Features: {features.shape}")
    print(f"Predictions: {predictions.shape}")
    
    # Save combined file
    np.savez(f'bilinear_features_block_{block_num}_combined.npz',
             features=features,
             predictions=predictions)

if __name__ == "__main__":
    block_num = 0  # Change this to combine different blocks
    combine_feature_files(1)
    combine_feature_files(2)
    combine_feature_files(3)
    combine_feature_files(4)  
    combine_feature_files(5)
    combine_feature_files(6)
    combine_feature_files(7)
