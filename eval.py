"""Script for evaluating a trained Sudoku model's performance."""

import functools
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf
from flax import jax_utils
from flax.training import checkpoints

from train import data
from train import evaluater
from train import model
from train import trainer

def get_dataset_size(file_path):
    """Get the number of puzzles in the dataset."""
    with tf.io.gfile.GFile(file_path, "rb") as f:
        data = np.load(f)
    return len(data)

def evaluate_model():
    """Evaluates a trained model on test data."""
    
    # Specify paths
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    test_data_path = "/home/groups/deissero/mrohatgi/mt/llm-reasoning-logic-puzzles/sudoku-code/datasets/test_sudoku_puzzles.npy"
    workdir = "eval_results"
    
    # Get total number of puzzles
    total_puzzles = get_dataset_size(test_data_path)
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
        minibatch_size = 64
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
    
    # Create dataset iterator
    eval_data_iter = data.create_iter(config, config.minibatch_size, train=False)
    
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
    
    # Create summary writer
    tf_summary_writer = tf.summary.create_file_writer(workdir)
    
    # Replicate state
    state = jax_utils.replicate(state)
    
    # Create parallel evaluation function
    p_eval_step = jax.pmap(
        functools.partial(
            evaluater.eval_step,
            config=model_config.replace(deterministic=True)
        ),
        axis_name="batch"
    )
    
    # Track metrics across all batches
    total_correct_cells = 0
    total_cells = 0
    total_correct_puzzles = 0
    total_processed = 0
    
    # Process all puzzles
    while total_processed < total_puzzles:
        # Get next batch
        batch_tuple = next(eval_data_iter)
        input_seq = np.array(batch_tuple[0])
        puzzle_sol = np.array(batch_tuple[1])
        start_index = np.array(batch_tuple[2])
        
        min_start_index = int(np.min(start_index))
        cur_input_seq = input_seq[:, :(min_start_index*3)]
        
        # Get device count for proper shaping
        n_devices = jax.local_device_count()
        per_device_batch_size = cur_input_seq.shape[0] // n_devices
        batch_size = per_device_batch_size * n_devices
        
        # Process sequence
        for i in range(min_start_index * 3, config.seq_len):
            padding = np.zeros((batch_size, config.seq_len - len(cur_input_seq[0])), dtype=np.int32)
            concat_batch = np.hstack((cur_input_seq, padding))
            
            # Prepare for multi-device processing
            device_batch = concat_batch.reshape(n_devices, per_device_batch_size, -1)
            device_batch = jax.tree_util.tree_map(np.asarray, device_batch)
            
            # Get predictions
            pred_logits = p_eval_step(state, device_batch)
            pred_logits = np.array(pred_logits).reshape(-1, config.seq_len, config.vocab_size)
            
            if i % 3 == 2:  # Value prediction
                max_number = pred_logits[:, i-1, :].argmax(axis=-1)
                mask_arr = np.array(i >= (3 * start_index)).squeeze()[:batch_size]
                next_number = max_number * mask_arr + (1 - mask_arr) * input_seq[:batch_size, i]
                
                # Verify predictions
                for k in range(len(cur_input_seq)):
                    if mask_arr[k]:
                        total_cells += 1
                        try:
                            evaluater.verify_sudoku_board(
                                puzzle_sol[k],
                                int(cur_input_seq[k][i-2]),
                                int(cur_input_seq[k][i-1]),
                                int(next_number[k])
                            )
                            total_correct_cells += 1
                        except AssertionError:
                            pass
            else:  # Row/column prediction
                max_pos = pred_logits[:, i-1, :].argmax(axis=-1)
                mask_arr = (i >= (3 * start_index)).squeeze()[:batch_size]
                next_number = max_pos * mask_arr + (1 - mask_arr) * input_seq[:batch_size, i]
            
            cur_input_seq = np.hstack((cur_input_seq[:batch_size], next_number.reshape(-1, 1)))
        
        # Check complete solutions
        remaining = total_puzzles - total_processed
        puzzles_to_use = min(batch_size, remaining)
        
        for i in range(puzzles_to_use):
            if evaluater.valid_solution(cur_input_seq[i]):
                total_correct_puzzles += 1
        
        total_processed += puzzles_to_use
        print(f"Processed {total_processed}/{total_puzzles} puzzles...", flush=True)
    
    # Calculate final metrics
    cell_accuracy = total_correct_cells / total_cells if total_cells > 0 else 0
    puzzle_accuracy = total_correct_puzzles / total_puzzles
    
    eval_metrics = {
        "acc": [cell_accuracy],
        "acc_complete_puzzle": [puzzle_accuracy]
    }
    
    # Log results
    print("\nFinal Results:")
    print(f"Total puzzles evaluated: {total_puzzles}")
    print(f"Correct cells: {total_correct_cells}/{total_cells} ({cell_accuracy:.2%})")
    print(f"Complete puzzles solved: {total_correct_puzzles}/{total_puzzles} ({puzzle_accuracy:.2%})")
    
    for key, value in eval_metrics.items():
        with tf_summary_writer.as_default():
            tf.summary.scalar(f"eval_{key}", value[0], step=0)
    
    return eval_metrics

if __name__ == '__main__':
    evaluate_model()
