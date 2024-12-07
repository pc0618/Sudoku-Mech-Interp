import functools
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import jax_utils
from flax.training import checkpoints
import pprint

def inspect_bilinear_matrices(checkpoint_path):
    """Loads model and extracts BilinearDense matrices."""
    
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
        ckpt_loc = checkpoint_path
    
    config = Config()
    
    # Initialize model configuration
    from train import model
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
    
    # Initialize model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, dropout_rng = random.split(rng, num=3)
    input_shape = (config.minibatch_size, config.seq_len)
    net = model.TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    initial_variables = net.init(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    # Create and restore state
    state = checkpoints.restore_checkpoint(config.ckpt_loc, initial_variables)
    if state is None:
        raise ValueError(f"Could not load checkpoint from {config.ckpt_loc}")
    print(f"\nRestored model from {config.ckpt_loc}")
    
    # Extract BilinearDense matrices
    bilinear_matrices = {}
    
    def extract_bilinear_params(params, path=""):
        for key, value in params.items():
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                extract_bilinear_params(value, current_path)
            elif isinstance(value, jnp.ndarray) and "BilinearDense_0" in current_path and "kernel" in current_path:
                # Split the kernel into left and right matrices
                w_l, w_r = jnp.split(value, 2, axis=1)
                layer_name = current_path.split("/")[-3]  # Get the transformer layer name
                bilinear_matrices[f"{layer_name}_left"] = w_l
                bilinear_matrices[f"{layer_name}_right"] = w_r
                print(f"\nFound BilinearDense matrices in {current_path}")
                print(f"Left matrix shape: {w_l.shape}")
                print(f"Right matrix shape: {w_r.shape}")
    
    # Extract matrices from params
    extract_bilinear_params(state['params'])
    
    return bilinear_matrices

# Example usage:
if __name__ == "__main__":
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    matrices = inspect_bilinear_matrices(checkpoint_path)
    
    # Print shapes and basic statistics for each matrix
    for name, matrix in matrices.items():
        np.save(name + '.npy', np.array(matrix))
        print(f"\nMatrix: {name}")
        print(f"Shape: {matrix.shape}")
        print(f"Mean: {jnp.mean(matrix):.6f}")
        print(f"Std: {jnp.std(matrix):.6f}")
        print(f"Min: {jnp.min(matrix):.6f}")
        print(f"Max: {jnp.max(matrix):.6f}")
