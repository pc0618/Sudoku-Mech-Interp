import functools
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf
from flax import jax_utils
from flax.training import checkpoints

from train import data
from train import model
from train import trainer

def get_dataset_size(file_path):
    """Get the number of puzzles in the dataset."""
    with tf.io.gfile.GFile(file_path, "rb") as f:
        data = np.load(f)
    return len(data)

def modified_eval_step(state, batch, config, block_num=0):
    """Modified eval step that returns layer outputs from specified block."""
    def _forward(params):
        variables = {'params': params, 'intermediates': {}}
        output, updated_vars = model.TransformerLMHeadModel(config).apply(
            variables,
            batch,
            capture_intermediates=True,
            mutable=['intermediates']
        )
        # Print intermediate keys for debugging
        print("Available intermediate keys:", updated_vars['intermediates'].keys())
        
        # Extract layer output for specified block
        layer_output = updated_vars['intermediates'][f'layer_{block_num}_output']
        if isinstance(layer_output, list):
            layer_output = layer_output[0]  # Take first element if it's a list
        if isinstance(layer_output, tuple):
            # Take the first element of the tuple which should be the actual output
            layer_output = layer_output[0]
            
        print(f"Shape of layer {block_num} output:", layer_output.shape)
        return output, layer_output
    
    return _forward(state.params)
    
def save_sequence_features(features, predictions, file_num, block_num):
    """Save sequence of features and predictions."""
    np.savez(f'bilinear_features_block_{block_num}_{file_num}.npz',
             features=features,
             predictions=predictions
            )
    print(f"Saved batch {file_num} with features shape {features.shape} and predictions shape {predictions.shape}")

def pad_features(features, target_length=243, feature_dim=576):
    """Pad features array to target length with zeros."""
    current_length = len(features)
    if current_length >= target_length:
        return features[:target_length]
    
    padding_length = target_length - current_length
    padding = np.zeros((padding_length, feature_dim))
    return np.vstack([features, padding])

def load_ordered_puzzles(file_path):
    """Load puzzles in original order and prepare them for processing."""
    with tf.io.gfile.GFile(file_path, "rb") as f:
        inputs_with_start_index = np.load(f)
    
    start_index = inputs_with_start_index[:, 0]  # Get the start index
    inputs = inputs_with_start_index[:, 1:]  
    
    # Delete the column corresponding to the set of strategies
    inputs = np.delete(inputs, np.arange(81) * 4 + 3, axis=1)
    
    return inputs, start_index.reshape(-1, 1)

def evaluate_and_collect_features(block_num=0):
    """Extract and save pre-logit features and predictions in sequence order."""
    
    # Specify paths
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    test_data_path = "/home/groups/deissero/mrohatgi/mt/llm-reasoning-logic-puzzles/sudoku-code/correct_puzzles.npy"
    
    # Load puzzles in original order
    all_inputs, all_start_indices = load_ordered_puzzles(test_data_path)
    total_puzzles = len(all_inputs)
    print(f"Found {total_puzzles} puzzles in dataset")
    
    # Model Configuration
    class Config:
        dataset = 'sudoku'
        seq_order = "solver-order"
        block_size = 81
        seq_len = 3 * block_size
        vocab_size = 11
        num_heads = 8
        num_layers = 8
        emb_dim = 576
        qkv_dim = 576
        mlp_dim = 6 * emb_dim
        minibatch_size = 32  # Reduced batch size for single GPU
        eval_epochs = 5
        max_steps = 2**22
        learning_rate = 0.0002
        end_lr_factor = 0.2
        warmup_tokens = 10000
        weight_decay = 0.005
        dtype = jnp.float32
        seed = 7
        test_puzzle_path = test_data_path
        ckpt_loc = checkpoint_path
    
    config = Config()
    
    # Hide GPUs from TensorFlow
    tf.config.experimental.set_visible_devices([], 'GPU')
    
    # Initialize model configuration
    model_config = model.TransformerConfig(
        dtype=config.dtype,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        emb_dim=config.emb_dim,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        deterministic=True,
    )
    
    print("Model configuration:", str(model_config.__dict__), flush=True)
    
    # Initialize model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, dropout_rng = random.split(rng, num=3)
    input_shape = (config.minibatch_size, config.seq_len)
    net = model.TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    initial_variables = net.init(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    # Create and restore state
    state, _ = trainer.get_state(config, net, initial_variables)
    state = checkpoints.restore_checkpoint(config.ckpt_loc, state)
    if state is None:
        raise ValueError(f"Could not load checkpoint from {config.ckpt_loc}")
    print(f"----------Restored model from {config.ckpt_loc} -----------")
    
    # Create evaluation function
    eval_step = functools.partial(modified_eval_step, config=model_config, block_num=block_num)
    
    # Process variables
    total_processed = 0
    file_count = 0
    batch_size = config.minibatch_size
    
    # Accumulate batches before saving
    accumulated_features = []
    accumulated_predictions = []
    puzzles_per_save = 20 * batch_size  # Save every 20 batches (640 puzzles)
    
    print(f"Starting puzzle processing for block {block_num}...")
    
    try:
        # Process puzzles in order
        for batch_start in range(0, total_puzzles, batch_size):
            try:
                # Get batch of puzzles
                batch_end = min(batch_start + batch_size, total_puzzles)
                current_batch_size = batch_end - batch_start
                
                input_seq = all_inputs[batch_start:batch_end]
                start_index = all_start_indices[batch_start:batch_end]
                
                # Process the entire batch at once first
                model_output = eval_step(state, input_seq)
                logits, features = model_output
                
                # Convert to numpy arrays
                logits = np.array(logits)
                features = np.array(features)
                
                # Remove the first dimension if needed
                if len(features.shape) == 4:
                    features = features.squeeze(0)
                
                # Initialize arrays for this batch
                batch_features = []
                batch_predictions = []
                
                # Process each puzzle
                for puzzle_idx in range(current_batch_size):
                    puzzle_start = int(start_index[puzzle_idx][0])
                    
                    # Extract features and predictions for this puzzle
                    puzzle_features = []
                    puzzle_predictions = []
                    
                    # Only collect features and predictions after the start index
                    valid_indices = range(puzzle_start * 3, config.seq_len)
                    for i in valid_indices:
                        next_prediction = logits[puzzle_idx, i-1, :].argmax()
                        curr_features = features[puzzle_idx, i-1, :]
                        
                        puzzle_features.append(curr_features)
                        puzzle_predictions.append(next_prediction)
                    
                    # Convert to arrays and pad
                    puzzle_features = np.array(puzzle_features)
                    puzzle_features = pad_features(puzzle_features)  # Pad to 243 length
                    puzzle_predictions = np.array(puzzle_predictions)
                    puzzle_predictions = np.pad(puzzle_predictions, 
                                             (0, 243 - len(puzzle_predictions)),
                                             'constant',
                                             constant_values=0)
                    
                    batch_features.append(puzzle_features)
                    batch_predictions.append(puzzle_predictions)
                
                # Convert batch lists to arrays
                batch_features = np.array(batch_features)
                batch_predictions = np.array(batch_predictions)
                
                # Add to accumulated arrays
                accumulated_features.extend(batch_features)
                accumulated_predictions.extend(batch_predictions)
                
                total_processed += current_batch_size
                print(f"Processed {total_processed}/{total_puzzles} puzzles...")
                
                # Save if we've accumulated enough puzzles or if we're at the end
                if len(accumulated_features) >= puzzles_per_save or total_processed >= total_puzzles:
                    accumulated_features_array = np.array(accumulated_features)
                    accumulated_predictions_array = np.array(accumulated_predictions)
                    save_sequence_features(accumulated_features_array, accumulated_predictions_array, file_count, block_num)
                    file_count += 1
                    accumulated_features = []
                    accumulated_predictions = []
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print(f"Current shapes:")
                if 'logits' in locals():
                    print(f"Logits: {logits.shape}")
                if 'features' in locals():
                    print(f"Features: {features.shape}")
                if 'puzzle_start' in locals():
                    print(f"Puzzle start: {puzzle_start}")
                if 'i' in locals():
                    print(f"Current index: {i}")
                raise e
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save any accumulated data
        if accumulated_features:
            try:
                accumulated_features_array = np.array(accumulated_features)
                accumulated_predictions_array = np.array(accumulated_predictions)
                save_sequence_features(accumulated_features_array, accumulated_predictions_array, file_count, block_num)
                file_count += 1
            except Exception as save_error:
                print(f"Error saving accumulated data: {str(save_error)}")
    
    print(f"\nCollection complete for block {block_num}!")
    print(f"Saved {total_puzzles} puzzles across {file_count} files")

if __name__ == '__main__':
    block_num = 1  # Change this to extract from different blocks (0-7)
    evaluate_and_collect_features(block_num)
    evaluate_and_collect_features(2)
    evaluate_and_collect_features(3)
    evaluate_and_collect_features(4)
    evaluate_and_collect_features(5)
    evaluate_and_collect_features(6)
    evaluate_and_collect_features(7)
