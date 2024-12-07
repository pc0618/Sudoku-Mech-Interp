import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
import pickle
import gc
from flax.training import checkpoints
from pathlib import Path

def get_mlp_down_projections(state):
    """
    Extract down projection matrices and convert to bfloat16.
    """
    down_projections = {}
    params = state['params']
    
    for block_idx in range(8):
        block_name = f'TransformerBlock_{block_idx}'
        down_proj = params[block_name]['BilinearMLP_0']['Dense_0']['kernel']
        down_proj = down_proj.astype(jnp.bfloat16)
        down_projections[block_idx] = down_proj
        print(f"Block {block_idx} down projection shape: {down_proj.shape}, dtype: {down_proj.dtype}")
        
    return down_projections

@jax.jit
def compute_symmetric_contribution(w_row, v_row):
    """
    Compute symmetric outer product for a single row.
    """
    outer = jnp.outer(w_row, v_row)
    return 0.5 * (outer + outer.T)

@jax.jit
def compute_symmetric_B_batched(W, V, batch_size=5000):
    """
    Compute symmetric B tensor using JAX operations.
    """
    W = W.T
    V = V.T
    d_output, d_input = W.shape
    
    B_flat = []
    for start_idx in range(0, d_output, batch_size):
        end_idx = min(start_idx + batch_size, d_output)
        print(f"Processing rows {start_idx} to {end_idx}")
        
        W_batch = W[start_idx:end_idx]
        V_batch = V[start_idx:end_idx]
        batch_contributions = vmap(compute_symmetric_contribution)(W_batch, V_batch)
        B_flat.append(batch_contributions)
    
    B = jnp.concatenate(B_flat, axis=0)
    B = jnp.transpose(B, (1, 2, 0))
    
    return B

def compute_projected_B_full(W, V, P):
    """
    Compute B tensor and project it with P.
    """
    print("Computing symmetric B tensor...")
    B = compute_symmetric_B_batched(W, V)
    print(f"B tensor shape: {B.shape}")
    del W, V
    gc.collect()
    
    print("Projecting with down projection matrix...")
    B_projected = jnp.tensordot(B, P, axes=[[2], [0]])
    print(f"Projected B tensor shape: {B_projected.shape}")
    del B, P
    gc.collect()
    
    return B_projected

def prepare_matrix_for_svd(B_projected):
    """
    Prepare matrix for full SVD computation.
    """
    B_2d = jnp.transpose(B_projected, (0, 1, 2))
    B_2d = B_2d.reshape(-1, B_projected.shape[2])
    print(f"Reshaped to 2D matrix: {B_2d.shape}")
    return B_2d

def full_svd(B_2d, k=None):
    """
    Compute full SVD and optionally truncate to k components.
    """
    print(f"Computing full SVD for matrix of shape {B_2d.shape}")
    
    U, s, Vh = jnp.linalg.svd(B_2d, full_matrices=False)
    
    if k is not None:
        U = U[:, :k]
        s = s[:k]
        Vh = Vh[:k, :]
        
    return U, s, Vh

def verify_svd(U, s, Vh, original_matrix, rtol=1e-3, atol=1e-3):
    """
    Verify the correctness of SVD decomposition.
    """
    U_orthonormal = jnp.allclose(
        U.T @ U, 
        jnp.eye(U.shape[1]), 
        rtol=rtol, 
        atol=atol
    )
    
    V_orthonormal = jnp.allclose(
        Vh @ Vh.T, 
        jnp.eye(Vh.shape[0]), 
        rtol=rtol, 
        atol=atol
    )
    
    reconstructed = U @ jnp.diag(s) @ Vh
    reconstruction_error = jnp.linalg.norm(original_matrix - reconstructed) / jnp.linalg.norm(original_matrix)
    
    s_sorted = jnp.all(s[:-1] >= s[1:])
    
    return {
        'U_orthonormal': bool(U_orthonormal),
        'V_orthonormal': bool(V_orthonormal),
        'singular_values_sorted': bool(s_sorted),
        'reconstruction_error': float(reconstruction_error),
        'max_singular_value': float(s[0]),
        'min_singular_value': float(s[-1]),
        'condition_number': float(s[0] / s[-1])
    }

def analyze_layer_full_svd(matrices_by_layer, down_projs, k=50, batch_size=16):
    """
    Analyze each layer using full SVD with verification.
    """
    results = {}
    
    for layer_num in matrices_by_layer:
        if layer_num != 7:
            continue
            
        print(f"\nAnalyzing layer {layer_num}")
        
        W = jnp.array(matrices_by_layer[layer_num]['left'], dtype=jnp.bfloat16)
        V = jnp.array(matrices_by_layer[layer_num]['right'], dtype=jnp.bfloat16)
        P = jnp.array(down_projs[layer_num], dtype=jnp.bfloat16)
        
        print(f"Converted matrices to bfloat16. Shapes: W {W.shape}, V {V.shape}, P {P.shape}")
        
        B_projected = compute_projected_B_full(W, V, P)
        del W, V, P
        gc.collect()
        
        B_2d = prepare_matrix_for_svd(B_projected)
        B_2d = B_2d.astype(jnp.float32)
        del B_projected
        gc.collect()
        
        U, s, Vh = full_svd(B_2d, k)
        
        verification_results = verify_svd(U, s, Vh, B_2d)
        del B_2d
        gc.collect()
        
        results[layer_num] = {
            'interaction_matrices': jnp.array(U, dtype=jnp.float32),
            'singular_values': jnp.array(s, dtype=jnp.float32),
            'output_directions': jnp.array(Vh, dtype=jnp.float32),
            'projected_dim': Vh.shape[1],
            'verification': verification_results
        }
        
        print(f"\nLayer {layer_num} Results:")
        print(f"Top 5 singular values: {results[layer_num]['singular_values'][:5]}")
        print(f"Verification results:")
        for key, value in verification_results.items():
            print(f"  {key}: {value}")
        
        del U, s, Vh
        gc.collect()
    
    return results

def save_results(results, save_dir='hosvd_results'):
    """
    Save all matrices and results locally.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save results as pickle
    print(f"\nSaving full results to {save_dir}/hosvd_results.pkl")
    with open(save_path / 'hosvd_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save individual matrices and verification results
    for layer_num, layer_results in results.items():
        layer_dir = save_path / f'layer_{layer_num}'
        layer_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving matrices for layer {layer_num}")
        
        np.save(
            layer_dir / 'interaction_matrices.npy',
            np.array(layer_results['interaction_matrices'])
        )
        
        np.save(
            layer_dir / 'singular_values.npy',
            np.array(layer_results['singular_values'])
        )
        
        np.save(
            layer_dir / 'output_directions.npy',
            np.array(layer_results['output_directions'])
        )
        
        # Save verification results
        metadata = {
            'interaction_matrices_shape': layer_results['interaction_matrices'].shape,
            'singular_values_shape': layer_results['singular_values'].shape,
            'output_directions_shape': layer_results['output_directions'].shape,
            'projected_dim': layer_results['projected_dim'],
            'top_5_singular_values': layer_results['singular_values'][:5].tolist(),
            'verification_results': layer_results['verification']
        }
        
        with open(layer_dir / 'metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

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
    
    # Run analysis with full SVD and verification
    results = analyze_layer_full_svd(
        matrices_by_layer,
        down_projs,
        k=50,
        batch_size=32
    )
    
    # Save results including verification metrics
    save_results(results, save_dir='full_svd_results')
    
    # Print summary of results and verification
    for layer_num, layer_results in results.items():
        print(f"\nLayer {layer_num} Final Summary:")
        print(f"Interaction matrices shape: {layer_results['interaction_matrices'].shape}")
        print(f"Top 5 singular values: {layer_results['singular_values'][:5]}")
        print(f"Output directions shape: {layer_results['output_directions'].shape}")
        print("\nVerification Results:")
        for key, value in layer_results['verification'].items():
            print(f"  {key}: {value}")
