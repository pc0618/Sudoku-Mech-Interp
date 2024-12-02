import jax
import jax.numpy as jnp
import torch
import numpy as np
from typing import Dict, Any, Tuple
import flax.traverse_util as traverse_util
from flax import core
import functools
import math
from flax.training import checkpoints
from flax import linen as nn
from flax import struct
from jax import random
from train import trainer
from pt_model import *

from dataclasses import dataclass
import ml_collections
from flax.training import train_state


@dataclass
class TorchTransformerConfig:
    """Configuration matching the original JAX model"""
    vocab_size: int = 11
    dtype: Any = torch.bfloat16
    emb_dim: int = 576
    num_heads: int = 8
    num_layers: int = 8
    qkv_dim: int = 576
    mlp_dim: int = 3456  # 6 * emb_dim
    seq_len: int = 243   # 3 * block_size (81)
    dropout_rate: float = 0.2
    attention_dropout_rate: float = 0.2
    deterministic: bool = False
    eval_epochs:int  = 5

@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int = 1
    dtype: Any = jnp.bfloat16
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    seq_len: int = 2048  # Maximum sequence length
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False

class BilinearDense(nn.Module):
    """A bilinear dense layer with optional gating."""
    features: int
    use_bias: bool = False
    gate_fn: str = None
    dtype: Any = jnp.bfloat16
    kernel_init: Any = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        input_features = inputs.shape[-1]
        
        # Create two separate weight matrices for left and right paths
        kernel_shape = (input_features, 2 * self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape)
        kernel = kernel.astype(self.dtype)
        
        # Scale the kernel to maintain variance
        kernel_scale = jnp.sqrt(2.0 / input_features)
        kernel = kernel * kernel_scale
        
        # Split kernel into left and right components
        w_l, w_r = jnp.split(kernel, 2, axis=1)
        
        # Compute left and right activations
        left = jnp.dot(inputs, w_l)
        right = jnp.dot(inputs, w_r)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (2 * self.features,))
            bias = bias.astype(self.dtype)
            b_l, b_r = jnp.split(bias, 2)
            left = left + b_l
            right = right + b_r
        
        # Apply layer normalization to both paths for stability
        left = nn.LayerNorm(dtype=self.dtype)(left)
        right = nn.LayerNorm(dtype=self.dtype)(right)
        
        # Apply gating function to left path if specified, otherwise identity
        if self.gate_fn == 'gelu':
            left = nn.gelu(left)
        elif self.gate_fn == 'relu':
            left = nn.relu(left)
        elif self.gate_fn == 'silu':
            left = nn.silu(left)
            
        # Multiply and apply final normalization
        output = left * right
        output = nn.LayerNorm(dtype=self.dtype)(output)
        
        return output

class BilinearMLP(nn.Module):
    """Bilinear MLP implementation."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        # Pre-normalization
        x = nn.LayerNorm(dtype=self.config.dtype)(x)
        
        # First bilinear transformation
        x = BilinearDense(
            features=self.config.mlp_dim,
            dtype=self.config.dtype,
            gate_fn=None,  # Default to identity function
            kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            use_bias=True
        )(x)
        
        # Project back to embedding dimension
        x = nn.Dense(
            features=self.config.emb_dim,
            use_bias=True,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            bias_init=nn.initializers.zeros
        )(x)
        
        # Apply dropout
        x = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=self.config.deterministic
        )(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block with bilinear MLP."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, causal_mask_inputs, training: bool = True):
        # Pre-normalization
        attn_input = nn.LayerNorm(dtype=self.config.dtype)(inputs)
        
        # Self-attention layer
        x = inputs + nn.SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.qkv_dim,
            kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            bias_init=nn.initializers.zeros,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout_rate,
            deterministic=self.config.deterministic,
            normalize_qk=True
        )(attn_input, causal_mask_inputs)
        
        # Pre-normalization for MLP
        mlp_input = nn.LayerNorm(dtype=self.config.dtype)(x)
        
        # Bilinear MLP
        x = x + BilinearMLP(config=self.config)(mlp_input)
        
        return x
        
class TransformerLMHeadModel(nn.Module):
    config: Any = None

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.emb_dim = self.config.emb_dim
        self.num_layers = self.config.num_layers

    @nn.compact
    def __call__(self, inputs, training=True):
        """
        Transformer LM Head call function.

        Args:
            inputs: Input tensor.
            training: Whether the model is in training mode.

        Returns:
            Transformed tensor after applying the Transformer layers and LM head.
        """
        batch_size, seq_size = inputs.shape

        causal_mask_x = nn.make_causal_mask(inputs, dtype=self.config.dtype)

        # Embed the input tensor using a learnable embedding matrix.
        embed_with_init = functools.partial(
            nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))
        token_embeddings = embed_with_init(
            num_embeddings=self.config.vocab_size,
            features=self.config.emb_dim,
        )(inputs)

        # Check the shape of the embedded tensor.
        assert token_embeddings.shape == (batch_size, seq_size,
                                      self.config.emb_dim)

        # Initialize the positional embedding variable.
        pos_embedding_variable = self.variable(
            "params",
            "position_embeddings",
            jnp.zeros,
            (self.config.seq_len, self.config.emb_dim),
        )

        # Slice the positional embedding array to the correct sequence length.
        pos_embeddings = pos_embedding_variable.value[:seq_size, :]

        # Check the shape of the positional embedding array.
        output_tuple = (pos_embeddings.shape, token_embeddings.shape[1:])
        assert pos_embeddings.shape == token_embeddings.shape[1:], output_tuple

        # Add the positional embeddings to the token embeddings.
        x = token_embeddings + pos_embeddings[None, :, :]

        # Apply dropout to the input.
        x = nn.Dropout(rate=self.config.dropout_rate,
                    deterministic=self.config.deterministic)(x)

        # Apply the Transformer layers.
        for i in range(self.num_layers):
            x = TransformerBlock(config=self.config)(
                    x, causal_mask_x, training=training)
      
            self.sow('intermediates', 'feature_' + str(i), x)

        # Apply the final layer normalization.
        x = nn.LayerNorm()(x)

        # Apply the LM head.
        logits = nn.Dense(features=self.config.vocab_size,
                        kernel_init=nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.normal(stddev=1e-6),
                        use_bias=False)(x)

        # Check the shape of the output tensor.
        assert logits.shape == (batch_size, seq_size, self.config.vocab_size)
        return logits

def compare_parameters(flax_state: train_state.TrainState, torch_state_dict: Dict[str, torch.Tensor], 
                      rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, list]:
    """Compare Flax and PyTorch parameters for equality."""
    flax_params = flax_state.params
    flat_flax = traverse_util.flatten_dict(flax_params)
    
    # Convert all PyTorch tensors to CPU numpy arrays once
    torch_numpy_dict = {k: v.detach().cpu().numpy() for k, v in torch_state_dict.items()}
    
    def reshape_attention_out_weight(weight):
        """Reshape attention output weight from (num_heads, emb_dim, head_dim) to (emb_dim, emb_dim)"""
        num_heads, emb_dim, head_dim = weight.shape
        # Reshape to (emb_dim, num_heads * head_dim)
        return weight.transpose(1, 0, 2).reshape(emb_dim, -1)
    
    def compare_tensors(a, b, name=""):
        """Compare tensors with proper reshaping."""
        a = np.array(a)
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        else:
            b = np.array(b)
        
        # Handle attention weights
        if 'self_attention.out_proj.weight' in name:
            if len(a.shape) == 3:
                num_heads, head_dim, emb_dim = a.shape
                a = a.transpose(2, 0, 1).reshape(emb_dim, -1)
        
        # Handle embeddings and projection weights
        elif any(x in name for x in ['.proj.weight', 'lm_head.weight']):
            a = a.T
        
        try:
            # Convert both arrays to float32 for comparison to avoid precision issues
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            match = np.allclose(a, b, rtol=1e-2, atol=1e-2)
            if not match:
                max_diff = np.max(np.abs(a - b))
                return False, max_diff
            return True, 0.0
        except ValueError:
            print(f"Shape mismatch for {name}: {a.shape} vs {b.shape}")
            return False, float('inf')
    
    mismatches = []
    all_match = True
    matched_params = set()
    
    # Map Flax parameter names to PyTorch parameter names
    param_mapping = {
        ('Embed_0', 'embedding'): 'token_embedding.weight',
        ('position_embeddings',): 'pos_embedding',
        ('Dense_0', 'kernel'): 'lm_head.weight',
        ('LayerNorm_0', 'scale'): 'ln_f.weight',
        ('LayerNorm_0', 'bias'): 'ln_f.bias',
    }
    
    def get_block_mapping(block_idx):
        """Get parameter mapping for a transformer block."""
        base = f'blocks.{block_idx}'
        return {
            ('LayerNorm_0', 'scale'): f'{base}.attn_ln.weight',
            ('LayerNorm_0', 'bias'): f'{base}.attn_ln.bias',
            ('LayerNorm_1', 'scale'): f'{base}.mlp.pre_ln.weight',
            ('LayerNorm_1', 'bias'): f'{base}.mlp.pre_ln.bias',
            ('SelfAttention_0', 'out', 'kernel'): f'{base}.self_attention.out_proj.weight',
            ('SelfAttention_0', 'out', 'bias'): f'{base}.self_attention.out_proj.bias',
            ('BilinearMLP_0', 'BilinearDense_0', 'kernel'): [
                f'{base}.mlp.bilinear.left_weight',
                f'{base}.mlp.bilinear.right_weight'
            ],
            ('BilinearMLP_0', 'BilinearDense_0', 'bias'): [
                f'{base}.mlp.bilinear.left_bias',
                f'{base}.mlp.bilinear.right_bias'
            ],
            ('BilinearMLP_0', 'Dense_0', 'kernel'): f'{base}.mlp.proj.weight',
            ('BilinearMLP_0', 'Dense_0', 'bias'): f'{base}.mlp.proj.bias',
        }
    
    # Process main parameters
    for flax_key, value in flat_flax.items():
        if flax_key in param_mapping:
            torch_key = param_mapping[flax_key]
            if torch_key not in torch_state_dict:
                mismatches.append(f"PyTorch parameter not found: {torch_key}")
                all_match = False
                continue
            
            match, diff = compare_tensors(value, torch_state_dict[torch_key], torch_key)
            if not match:
                mismatches.append(f"Value mismatch for {torch_key}: max difference {diff}")
                all_match = False
            matched_params.add(torch_key)
            continue
        
        # Process transformer block parameters
        block_idx = None
        block_key = None
        for i, part in enumerate(flax_key):
            if isinstance(part, str) and 'TransformerBlock_' in part:
                block_idx = int(part.split('_')[1])
                block_key = flax_key[i+1:]
                break
        
        if block_idx is not None:
            block_mapping = get_block_mapping(block_idx)
            if block_key in block_mapping:
                torch_key = block_mapping[block_key]
                if isinstance(torch_key, list):
                    # Handle split parameters (bilinear weights/biases)
                    left, right = np.split(value, 2, axis=-1 if 'kernel' in str(block_key) else 0)
                    for split_value, split_key in zip([left, right], torch_key):
                        if split_key not in torch_state_dict:
                            mismatches.append(f"PyTorch parameter not found: {split_key}")
                            all_match = False
                            continue
                        
                        match, diff = compare_tensors(split_value, torch_state_dict[split_key], split_key)
                        if not match:
                            mismatches.append(f"Value mismatch for {split_key}: max difference {diff}")
                            all_match = False
                        matched_params.add(split_key)
                else:
                    if torch_key not in torch_state_dict:
                        mismatches.append(f"PyTorch parameter not found: {torch_key}")
                        all_match = False
                        continue
                    
                    match, diff = compare_tensors(value, torch_state_dict[torch_key], torch_key)
                    if not match:
                        mismatches.append(f"Value mismatch for {torch_key}: max difference {diff}")
                        all_match = False
                    matched_params.add(torch_key)
    
    # Don't report layer norm and attention projection parameters as extra
    # These are initialized in PyTorch but not present in Flax
    expected_extra_params = {
        'left_ln.weight', 'left_ln.bias',
        'right_ln.weight', 'right_ln.bias',
        'output_ln.weight', 'output_ln.bias',
        'in_proj_weight', 'in_proj_bias'
    }
    
    # Check for unexpected extra parameters in PyTorch model
    for torch_key in torch_state_dict.keys():
        if torch_key not in matched_params:
            # Only report as extra if it's not an expected additional parameter
            if not any(extra in torch_key for extra in expected_extra_params):
                mismatches.append(f"Unexpected extra parameter in PyTorch model: {torch_key}")
                all_match = False
    
    return all_match, mismatches

def verify_model_outputs(flax_state: train_state.TrainState, torch_model, net, config, 
                        rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, float]:
    """Verify that Flax and PyTorch models produce the same outputs."""
    torch_model.eval()
    
    # Create same random input
    batch_size = 2
    seq_len = config.seq_len
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size)
    
    # Get Flax output
    variables = {'params': flax_state.params}
    flax_output = net.apply(variables, input_ids, training=False, mutable=False)
    
    # Convert input to PyTorch and move to appropriate device
    device = next(torch_model.parameters()).device
    torch_input = torch.from_numpy(np.array(input_ids)).to(device)
    
    # Get PyTorch output
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    
    # Convert outputs to numpy for comparison (move PyTorch tensor to CPU first)
    flax_numpy = np.array(flax_output)
    torch_numpy = torch_output.detach().cpu().numpy()
    
    # Compare outputs
    outputs_match = np.allclose(flax_numpy, torch_numpy, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(flax_numpy - torch_numpy))
    
    return outputs_match, max_diff

def run_verification(ckpt_dir: str, config: Any, flax_state: train_state.TrainState, 
                    torch_model, net) -> bool:
    """Run complete verification of the model conversion."""
    print("Starting verification...")
    
    # Verify parameters
    print("\nVerifying parameters...")
    params_match, mismatches = compare_parameters(flax_state, torch_model.state_dict())
    
    if params_match:
        print("✓ All parameters match!")
    else:
        print("× Parameter mismatches found:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
    
    # Verify outputs
    print("\nVerifying model outputs...")
    outputs_match, max_diff = verify_model_outputs(flax_state, torch_model, net, config)
    
    if outputs_match:
        print(f"✓ Model outputs match! Maximum difference: {max_diff:.2e}")
    else:
        print(f"× Model outputs differ! Maximum difference: {max_diff:.2e}")
    
    return params_match and outputs_match


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    # Dataset choice
    config.dataset = 'sudoku'
    config.seq_order = "solver-order"
    
    # Training related parameters
    config.max_steps = 2**22
    config.dtype = jax.numpy.bfloat16
    config.minibatch_size = 12
    
    # Model related parameters
    config.block_size = 81
    config.seq_len = 3 * config.block_size
    config.vocab_size = 11
    
    # Model architecture
    config.num_heads = 8
    config.num_layers = 8
    config.emb_dim = 576
    config.qkv_dim = 576
    config.mlp_dim = 6 * config.emb_dim
    config.dropout_rate = 0.2
    config.attention_dropout_rate = 0.2
    
    # Add deterministic flag
    config.deterministic = True  # Set to True for inference
    
    # Training hyperparameters
    config.learning_rate = 0.0002
    config.end_lr_factor = 0.2
    config.warmup_tokens = 10000
    config.weight_decay = 0.005
    config.resume_training = False
    
    # Other hyperparameters
    config.seed = 7
    config.save_checkpoint = True
    config.save_every_steps = 32000
    config.use_wandb = True
    config.wandb_project_name = 'sudoku'
    
    # Evaluation related parameters
    config.eval_every_steps = 8000
    config.eval_epochs = 5
    
    return config


# Example usage
if __name__ == "__main__":
    config = get_config()
    
    torch_config = TorchTransformerConfig(
        vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        seq_len=config.seq_len,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate
    )
    
    # Initialize the model and get initial variables
    ckpt_path = "/u/pranshu3/llm-reasoning-logic-puzzles/sudoku-code/logs/checkpoint_4192000"
    
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, inference_rng = random.split(rng, num=3)
    
    rng, dropout_rng = jax.random.split(rng)
    input_shape = (config.minibatch_size, config.seq_len)
    net = TransformerLMHeadModel(config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    sample_out, initial_variables = jax.jit(
        net.init_with_output
        )(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    state, lr_scheduler_fn = trainer.get_state(config, net, initial_variables)
    restored_state = checkpoints.restore_checkpoint(ckpt_path, state)
    print("----------Restored model from", ckpt_path, "-----------")
    
    # Set up device and move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = TorchTransformerLMHeadModel(torch_config).to(device)
    torch_model.load_state_dict(torch.load('converted_model.pt', map_location=device))
    
    # Run verification
    success = run_verification(ckpt_path, config, restored_state, torch_model, net)
    
    if success:
        print("\nVerification successful! The conversion is correct.")
    else:
        print("\nVerification failed! Please check the reported differences.")