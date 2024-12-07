import functools
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import jax_utils
from flax.training import checkpoints
import pprint

def analyze_bilinear_matrices(checkpoint_path):
    """Loads model and extracts BilinearDense matrices for analysis."""
    
    # Model Configuration (same as before)
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
    
    # Initialize model and load checkpoint
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, dropout_rng = random.split(rng, num=3)
    input_shape = (config.minibatch_size, config.seq_len)
    net = model.TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    initial_variables = net.init(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    state = checkpoints.restore_checkpoint(config.ckpt_loc, initial_variables)
    if state is None:
        raise ValueError(f"Could not load checkpoint from {config.ckpt_loc}")
    print(f"\nRestored model from {config.ckpt_loc}")
    
    # Dictionary to store matrices by layer
    matrices_by_layer = {}
    
    def extract_bilinear_params(params, path=""):
        for key, value in params.items():
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                extract_bilinear_params(value, current_path)
            elif isinstance(value, jnp.ndarray) and "BilinearDense_0" in current_path and "kernel" in current_path:
                # Convert to numpy and split the kernel
                kernel_np = np.array(value)
                w_l, w_r = np.split(kernel_np, 2, axis=1)
                
                # Extract layer number from path
                layer_parts = current_path.split("/")
                layer_idx = next(i for i, part in enumerate(layer_parts) if part.startswith("TransformerBlock_"))
                layer_num = int(layer_parts[layer_idx].split("_")[1])
                
                # Store matrices for this layer
                matrices_by_layer[layer_num] = {
                    'left': w_l,
                    'right': w_r,
                    'full_kernel': kernel_np,
                    'path': current_path
                }
                
                print(f"\nLayer {layer_num} BilinearDense matrices:")
                print(f"Left matrix shape: {w_l.shape}")
                print(f"Right matrix shape: {w_r.shape}")
    
    # Extract all matrices
    extract_bilinear_params(state['params'])
    
    # Example of how to access specific matrices
    def get_layer_matrices(layer_num):
        """Get matrices for a specific layer."""
        if layer_num not in matrices_by_layer:
            raise ValueError(f"Layer {layer_num} not found")
        return matrices_by_layer[layer_num]
    
    # Example analysis functions
    def analyze_layer(layer_num):
        """Analyze matrices from a specific layer."""
        matrices = get_layer_matrices(layer_num)
        
        # Basic statistics for left matrix
        left_stats = {
            'mean': np.mean(matrices['left']),
            'std': np.std(matrices['left']),
            'min': np.min(matrices['left']),
            'max': np.max(matrices['left']),
            'shape': matrices['left'].shape
        }
        
        # Basic statistics for right matrix
        right_stats = {
            'mean': np.mean(matrices['right']),
            'std': np.std(matrices['right']),
            'min': np.min(matrices['right']),
            'max': np.max(matrices['right']),
            'shape': matrices['right'].shape
        }
        
        return {'left_stats': left_stats, 'right_stats': right_stats}
    
    return matrices_by_layer, get_layer_matrices, analyze_layer

# Example usage:
if __name__ == "__main__":
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    matrices_by_layer, get_layer_matrices, analyze_layer = analyze_bilinear_matrices(checkpoint_path)
    
    # Example: Get matrices from layer 0
    layer_0_matrices = get_layer_matrices(0)
    left_matrix_layer_0 = layer_0_matrices['left']
    right_matrix_layer_0 = layer_0_matrices['right']
    
    # Example: Analyze specific layer
    layer_0_analysis = analyze_layer(0)
    print("\nLayer 0 Analysis:")
    print("Left matrix statistics:", layer_0_analysis['left_stats'])
    print("Right matrix statistics:", layer_0_analysis['right_stats'])
    
    # Example: Access any layer's matrices
    for layer_num in matrices_by_layer.keys():
        matrices = get_layer_matrices(layer_num)
        print(f"\nLayer {layer_num} shapes:")
        print(f"Left matrix: {matrices['left'].shape}")
        print(f"Right matrix: {matrices['right'].shape}")
        
        # Example further analysis
        left_matrix = matrices['left']
        right_matrix = matrices['right']
        
        # Compute singular values
        left_svd = np.linalg.svd(left_matrix, compute_uv=False)
        right_svd = np.linalg.svd(right_matrix, compute_uv=False)
        
        print(f"Top 5 singular values:")
        print(f"Left matrix: {left_svd[:5]}")
        print(f"Right matrix: {right_svd[:5]}")
