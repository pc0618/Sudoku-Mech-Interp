import numpy as np

def combine_feature_files(num_files=13):
    """Combine feature files while maintaining order."""
    
    # First pass: get total size and verify shapes
    total_puzzles = 0
    feature_dim = None
    pred_dim = None
    
    print("Verifying file shapes...")
    for i in range(num_files):
        data = np.load(f'sequence_features_{i}.npz')
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
        data = np.load(f'sequence_features_{i}.npz')
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
    print("\nSaving combined file...")
    np.savez('sequence_features_combined.npz', 
             features=features, 
             predictions=predictions)
    print("Done!")
    
    # Verify first and last puzzles of each file
    print("\nVerifying order preservation...")
    current_idx = 0
    for i in range(num_files):
        data = np.load(f'sequence_features_{i}.npz')
        n_puzzles = data['features'].shape[0]
        
        # Check first puzzle
        assert np.array_equal(data['features'][0], features[current_idx]), f"Mismatch at start of file {i}"
        # Check last puzzle
        assert np.array_equal(data['features'][-1], features[current_idx + n_puzzles - 1]), f"Mismatch at end of file {i}"
        
        current_idx += n_puzzles
    print("Order verification passed!")

if __name__ == '__main__':
    combine_feature_files()
