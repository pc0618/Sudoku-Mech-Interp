import jax
import jax.numpy as jnp
from jax import lax, vmap
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from flax.training import checkpoints
import numpy as np
import gc

def get_mlp_down_projections(state):
    """
    Extract down projection matrices and convert to bfloat16.
    """
    down_projections = {}
    params = state['params']
    
    for block_idx in range(8):
        block_name = f'TransformerBlock_{block_idx}'
        down_proj = params[block_name]['BilinearMLP_0']['Dense_0']['kernel']
        # Convert to bfloat16
        down_proj = down_proj.astype(jnp.bfloat16)
        down_projections[block_idx] = down_proj
        print(f"Block {block_idx} down projection shape: {down_proj.shape}, dtype: {down_proj.dtype}")
        
    return down_projections
    
@jax.jit
def compute_symmetric_contribution(w_row, v_row):
    """
    Compute symmetric outer product for a single row.
    Args:
        w_row: Single row from W matrix (d_output,)
        v_row: Single row from V matrix (d_output,)
    Returns:
        Symmetric contribution (d_input, d_input)
    """
    outer = jnp.outer(w_row, v_row)
    return 0.5 * (outer + outer.T)

@jax.jit
def compute_symmetric_B_batched(W, V, batch_size=32):
    """
    Compute symmetric B tensor using JAX operations.
    Args:
        W: Weight matrix (576, 3456)
        V: Weight matrix (576, 3456)
        batch_size: Batch size for processing
    Returns:
        B tensor with shape (576, 576, 3456)
    """
    # Transpose W and V to get (3456, 576)
    W = W.T
    V = V.T
    d_output, d_input = W.shape  # d_output=3456, d_input=576
    
    B_flat = []
    for start_idx in range(0, d_output, batch_size):
        end_idx = min(start_idx + batch_size, d_output)
        print(f"Processing rows {start_idx} to {end_idx}")
        
        # Get batch of rows
        W_batch = W[start_idx:end_idx]  # (batch_size, 576)
        V_batch = V[start_idx:end_idx]  # (batch_size, 576)
        
        # Compute symmetric contributions for batch
        batch_contributions = vmap(compute_symmetric_contribution)(W_batch, V_batch)  # (batch_size, 576, 576)
        B_flat.append(batch_contributions)
    
    # Concatenate all batches
    B = jnp.concatenate(B_flat, axis=0)  # (3456, 576, 576)
    
    # Transpose to get (576, 576, 3456)
    B = jnp.transpose(B, (1, 2, 0))
    
    return B

@jax.jit
def compute_projected_B(W, V, P):
    """
    Compute B tensor and project it with P.
    Args:
        W: Weight matrix (576, 3456)
        V: Weight matrix (576, 3456)
        P: Down projection matrix (3456, 576)
    Returns:
        Projected B tensor (576, 576, 576)
    """
    print("Computing symmetric B tensor...")
    B = compute_symmetric_B_batched(W, V)  # Shape: (576, 576, 3456)
    print(f"B tensor shape: {B.shape}")
    del W, V
    gc.collect()
    
    print("Projecting with down projection matrix...")
    B_projected = jnp.tensordot(B, P, axes=[[2], [0]])  # Shape: (576, 576, 576)
    print(f"Projected B tensor shape: {B_projected.shape}")
    del B, P
    gc.collect()
    
    return B_projected
    
def create_sparse_matrix(B_projected, threshold=1e-5):
    """
    Convert dense matrix to sparse format efficiently.
    Args:
        B_projected: Dense tensor of shape (576, 576, 576) representing (input, input, output)
        threshold: Values below this are set to 0
    Returns:
        Sparse matrix of shape (input*input, output) = (331776, 576)
    """
    print(f"Converting {B_projected.shape} tensor to sparse format with threshold {threshold}...")
    
    # Reshape to (input*input, output) = (576*576, 576)
    B_2d = jnp.transpose(B_projected, (0, 1, 2))  # Ensure dimensions are in right order
    B_2d = B_2d.reshape(-1, B_projected.shape[2])  # Reshape to (576*576, 576)
    print(f"Reshaped to 2D matrix: {B_2d.shape}")
    
    # Convert to numpy float32 for better numerical stability in sparse computation
    B_np = np.array(B_2d, dtype=np.float32)
    del B_projected, B_2d
    gc.collect()
    
    B_np[np.abs(B_np) < threshold] = 0
    B_sparse = sp.csr_matrix(B_np)
    del B_np
    gc.collect()
    
    print(f"Created sparse matrix with shape {B_sparse.shape}, sparsity: {B_sparse.nnz / np.prod(B_sparse.shape):.4%}")
    return B_sparse
    
def sparse_svd(B_sparse, k):
    """
    Compute SVD using only sparse representation.
    """
    print(f"Computing sparse SVD for matrix of shape {B_sparse.shape} with k={k}...")
    
    # k must be less than min(matrix.shape)
    k_safe = min(k, min(B_sparse.shape) - 1)
    
    # Use sparse SVD
    U, s, Vh = svds(B_sparse, k=k_safe)
    
    # Sort in descending order of singular values
    idx = np.argsort(s)[::-1]
    U = U[:, idx]
    s = s[idx]
    Vh = Vh[idx, :]
    
    return U, s, Vh

def analyze_layer_efficient(matrices_by_layer, down_projs, k=50, batch_size=16, threshold=0.05):
    """
    Analyze each layer using memory-efficient sparse HOSVD.
    """
    results = {}
    
    for layer_num in matrices_by_layer:
        print(f"\nAnalyzing layer {layer_num}")
        
        # Get matrices and convert to float16
        W = jnp.array(matrices_by_layer[layer_num]['left'], dtype=jnp.float16)
        V = jnp.array(matrices_by_layer[layer_num]['right'], dtype=jnp.float16)
        P = jnp.array(down_projs[layer_num], dtype=jnp.float16)
        
        print(f"Converted matrices to float16. Shapes: W {W.shape}, V {V.shape}, P {P.shape}")
        
        B_projected = compute_projected_B(W, V, P)
        del W, V, P
        gc.collect()
        
        B_sparse = create_sparse_matrix(B_projected, threshold)
        del B_projected
        gc.collect()
        
        U, s, Vh = sparse_svd(B_sparse, k)
        del B_sparse
        gc.collect()
        
        # Store results with new, more descriptive names
        results[layer_num] = {
            'interaction_matrices': jnp.array(U, dtype=jnp.float32),  # (331776, 50) - how input pairs interact
            'singular_values': jnp.array(s, dtype=jnp.float32),      # (50,) - strength of each pattern
            'output_directions': jnp.array(Vh, dtype=jnp.float32),   # (50, 576) - mapping to output dimensions
            'projected_dim': Vh.shape[1]
        }
        
        print(f"Top 5 singular values for layer {layer_num}:")
        print(results[layer_num]['singular_values'][:5])
        print(f"Interaction matrices shape: {results[layer_num]['interaction_matrices'].shape}")
        print(f"Output directions shape: {results[layer_num]['output_directions'].shape}")
        
        del U, s, Vh
        gc.collect()
    
    return results
    
def save_results(results, save_dir='hosvd_results'):
    """
    Save all matrices and results locally in multiple formats.
    Args:
        results: Dictionary containing results for each layer
        save_dir: Directory to save results in
    """
    import os
    import pickle
    import numpy as np
    from pathlib import Path
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results as pickle for easy loading in Python
    print(f"\nSaving full results to {save_dir}/hosvd_results.pkl")
    with open(save_path / 'hosvd_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save individual matrices for each layer
    for layer_num, layer_results in results.items():
        layer_dir = save_path / f'layer_{layer_num}'
        layer_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving matrices for layer {layer_num}")
        
        # Save interaction matrices
        np.save(
            layer_dir / 'interaction_matrices.npy',
            np.array(layer_results['interaction_matrices'])
        )
        
        # Save singular values
        np.save(
            layer_dir / 'singular_values.npy',
            np.array(layer_results['singular_values'])
        )
        
        # Save output directions
        np.save(
            layer_dir / 'output_directions.npy',
            np.array(layer_results['output_directions'])
        )
        
        # Save a metadata file with shapes and additional info
        metadata = {
            'interaction_matrices_shape': layer_results['interaction_matrices'].shape,
            'singular_values_shape': layer_results['singular_values'].shape,
            'output_directions_shape': layer_results['output_directions'].shape,
            'projected_dim': layer_results['projected_dim'],
            'top_5_singular_values': layer_results['singular_values'][:5].tolist()
        }
        
        with open(layer_dir / 'metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    print(f"\nSaved all results in {save_dir}/")
    
if __name__ == "__main__":
    # Set paths and config
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    
    # Initialize model configuration
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
        dtype = jnp.bfloat16
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
    rng, init_rng, dropout_rng = jax.random.split(rng, num=3)
    input_shape = (config.minibatch_size, config.seq_len)
    net = model.TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    initial_variables = net.init(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    state = checkpoints.restore_checkpoint(config.ckpt_loc, initial_variables)
    if state is None:
        raise ValueError(f"Could not load checkpoint from {config.ckpt_loc}")
    print(f"\nRestored model from {config.ckpt_loc}")
    
    # Get down projections and bilinear matrices
    down_projs = get_mlp_down_projections(state)
    
    matrices_by_layer = {}
    params = state['params']
    for layer_idx in range(8):
        layer_name = f'TransformerBlock_{layer_idx}'
        kernel = params[layer_name]['BilinearMLP_0']['BilinearDense_0']['kernel']
        w_l, w_r = jnp.split(kernel, 2, axis=1)
        matrices_by_layer[layer_idx] = {'left': w_l, 'right': w_r}
        print(f"Layer {layer_idx} matrices shapes: left {w_l.shape}, right {w_r.shape}")
    
    # Run analysis
    results = analyze_layer_efficient(
        matrices_by_layer,
        down_projs,
        k=50,
        batch_size=32
    )
    
    # Save all results
    save_results(results, save_dir='hosvd_results')
    
    # Print summary (as before)
    for layer_num, layer_results in results.items():
        print(f"\nLayer {layer_num} Results:")
        print(f"Interaction matrices shape: {layer_results['interaction_matrices'].shape}")
        print(f"Top 5 singular values: {layer_results['singular_values'][:5]}")
        print(f"Output directions shape: {layer_results['output_directions'].shape}")
