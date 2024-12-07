import numpy as np
from scipy import linalg

def compute_bilinear_tensor(W, V):
    """
    Compute the bilinear tensor B from left (W) and right (V) matrices.
    B[i] is the outer product of the i'th rows of W and V.
    Returns only the symmetric component.
    
    Args:
        W: Left weight matrix (d_hidden, d_input)
        V: Right weight matrix (d_hidden, d_input)
    
    Returns:
        B: Symmetric bilinear tensor (d_hidden, d_input, d_input)
    """
    d_hidden, d_input = W.shape
    B = np.zeros((d_hidden, d_input, d_input))
    
    # Compute outer product for each row
    for i in range(d_hidden):
        w_i = W[i]  # i'th row of W
        v_i = V[i]  # i'th row of V
        B_i = np.outer(w_i, v_i)
        
        # Keep only symmetric component: B_sym = 1/2(B + B^T)
        B[i] = 0.5 * (B_i + B_i.T)
    
    return B

def perform_hosvd(B):
    """
    Perform HOSVD on bilinear tensor B by flattening and SVD.
    
    Args:
        B: Bilinear tensor (d_hidden, d_input, d_input)
    
    Returns:
        output_vectors: U matrix from SVD (d_hidden, d_hidden)
        interaction_matrices: Reshaped V matrix (d_hidden, d_input, d_input)
        singular_values: Singular values
    """
    d_hidden, d_input, _ = B.shape
    
    # Flatten B into (d_hidden, d_input^2) matrix
    B_flat = B.reshape(d_hidden, d_input * d_input)
    
    # Perform SVD
    U, s, Vh = linalg.svd(B_flat, full_matrices=False)
    
    # Reshape Vh rows back into matrices
    interaction_matrices = Vh.reshape(-1, d_input, d_input)
    
    return U, interaction_matrices, s

def analyze_layer_directions(matrices_by_layer):
    """
    Analyze output directions for each layer using HOSVD.
    
    Args:
        matrices_by_layer: Dictionary with left/right matrices for each layer
        
    Returns:
        Dictionary of HOSVD results for each layer
    """
    results = {}
    
    for layer_num, layer_matrices in matrices_by_layer.items():
        print(f"\nAnalyzing layer {layer_num}")
        
        # Get left and right matrices
        W = layer_matrices['left']
        V = layer_matrices['right']
        
        # Compute bilinear tensor
        B = compute_bilinear_tensor(W, V)
        
        # Perform HOSVD
        output_vectors, interaction_matrices, singular_values = perform_hosvd(B)
        
        # Store results
        results[layer_num] = {
            'output_vectors': output_vectors,  # U matrix from SVD
            'interaction_matrices': interaction_matrices,  # Reshaped V matrix
            'singular_values': singular_values,
            'bilinear_tensor': B
        }
        
        # Print some basic statistics
        print(f"Shape of output vectors: {output_vectors.shape}")
        print(f"Number of interaction matrices: {len(interaction_matrices)}")
        print(f"Top 5 singular values: {singular_values[:5]}")
        
    return results

if __name__ == "__main__":
    # First get matrices using your existing code
    checkpoint_path = "/home/groups/deissero/mrohatgi/mt/checkpoint_4192000"
    from inspect_matrices2 import analyze_bilinear_matrices
    matrices_by_layer, _, _ = analyze_bilinear_matrices(checkpoint_path)
    
    # Perform HOSVD analysis
    hosvd_results = analyze_layer_directions(matrices_by_layer)
    
    # Example of accessing results for a specific layer
    layer_0_results = hosvd_results[0]
    
    # The output vectors U give the relevant output directions
    output_directions = layer_0_results['output_vectors']
    
    # The interaction matrices Q can be obtained from V
    interaction_matrices = layer_0_results['interaction_matrices']
    
    # Singular values indicate importance of each direction
    singular_values = layer_0_results['singular_values']
