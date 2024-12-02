import jax
import jax.numpy as jnp
import torch
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import functools
import math
import flax.traverse_util as traverse_util
from train import trainer
from flax.training import checkpoints
from flax import linen as nn
from flax import struct
from jax import random
from typing import Dict, Any, Callable, Optional
from flax import core
from dataclasses import dataclass
import ml_collections
from pt_model import *

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

def convert_flax_to_torch_weights(flax_state: Dict[str, Any], torch_model) -> None:
    """Convert Flax parameters to PyTorch parameters."""
    if isinstance(flax_state, core.FrozenDict):
        flax_params = flax_state['params']
    else:
        flax_params = flax_state.params

    flat_params = traverse_util.flatten_dict(flax_params)
    
    def to_numpy(x):
        """Convert JAX array to numpy array."""
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)
    """
    def set_param(module, name, value):
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            module = getattr(module, part)
        value = to_numpy(value)
        
        # Handle LayerNorm parameters
        if 'LayerNorm' in name:
            if 'scale' in name:
                param_name = 'weight'
                name_parts[-1] = param_name
            elif 'bias' in name:
                param_name = 'bias'
                name_parts[-1] = param_name
        
        # Handle attention output projection weights
        elif 'self_attention.out_proj.weight' in name:
            # Reshape from (num_heads, head_dim, emb_dim) to (emb_dim, emb_dim)
            num_heads, head_dim, emb_dim = value.shape
            value = value.transpose(2, 0, 1).reshape(emb_dim, -1)
        
        # Handle LM head and MLP projection weights
        elif any(x in name for x in ['lm_head.weight', '.proj.weight']):
            value = value.T
        
        # Handle embeddings
        elif name == 'token_embedding.weight':
            pass
        elif name == 'pos_embedding':
            # Already in (seq_len, emb_dim), just add batch dimension
            value = value[None, :, :]


        value_tensor = torch.from_numpy(value)
        setattr(module, name_parts[-1], torch.nn.Parameter(value_tensor))
    """
    def set_param(module, name, value):
        """Set parameter with proper shape handling."""
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            module = getattr(module, part)
        value = to_numpy(value)
        
        # Handle token embeddings
        if name == 'token_embedding.weight':
            # Flax: (emb_dim, vocab_size) -> PyTorch: (vocab_size, emb_dim)
            #value = value.T
            pass
        elif name == 'pos_embedding':
            # Already in (seq_len, emb_dim), just add batch dimension
            value = value[None, :, :]
        
        # Handle attention output projection weights
        elif 'self_attention.out_proj.weight' in name:
            if len(value.shape) == 3:
                # Reshape from (num_heads, head_dim, emb_dim) to (emb_dim, emb_dim)
                num_heads, head_dim, emb_dim = value.shape
                value = value.transpose(2, 0, 1).reshape(emb_dim, -1)
        
        # Handle MLP projection weights
        elif '.proj.weight' in name or 'lm_head.weight' in name:
            value = value.T
        
        # Handle LayerNorm parameters
        elif 'LayerNorm' in name or 'ln' in name:
            if 'scale' in name:
                name_parts[-1] = 'weight'
            elif 'bias' in name:
                name_parts[-1] = 'bias'
        
        # Convert to bfloat16 tensor
        value_tensor = torch.from_numpy(value).to(torch.bfloat16)
        setattr(module, name_parts[-1], torch.nn.Parameter(value_tensor))
    # Process main parameters
    main_params = {
        ('Embed_0', 'embedding'): 'token_embedding.weight',
        ('position_embeddings',): 'pos_embedding',
        ('Dense_0', 'kernel'): 'lm_head.weight',
        ('LayerNorm_0', 'scale'): 'ln_f.weight',
        ('LayerNorm_0', 'bias'): 'ln_f.bias',
    }

    # Process parameters
    for flax_key, value in flat_params.items():
        if flax_key in main_params:
            set_param(torch_model, main_params[flax_key], value)
            continue

        block_idx = None
        remaining_key = None
        for i, part in enumerate(flax_key):
            if isinstance(part, str) and 'TransformerBlock_' in part:
                block_idx = int(part.split('_')[1])
                remaining_key = flax_key[i+1:]
                break

        if block_idx is not None:
            base = f'blocks.{block_idx}'
            key_str = '/'.join(str(k) for k in remaining_key)

            # Handle BilinearMLP parameters
            if 'BilinearDense_0' in key_str:
                if 'kernel' in key_str:
                    left, right = np.split(to_numpy(value), 2, axis=1)
                    set_param(torch_model, f'{base}.mlp.bilinear.left_weight', left)
                    set_param(torch_model, f'{base}.mlp.bilinear.right_weight', right)
                elif 'bias' in key_str:
                    left, right = np.split(to_numpy(value), 2)
                    set_param(torch_model, f'{base}.mlp.bilinear.left_bias', left)
                    set_param(torch_model, f'{base}.mlp.bilinear.right_bias', right)
            
            # Handle attention parameters
            elif 'SelfAttention_0' in key_str:
                if 'out' in key_str:
                    param_name = 'weight' if 'kernel' in key_str else 'bias'
                    set_param(torch_model, f'{base}.self_attention.out_proj.{param_name}', value)
                
            # Handle layer norms
            elif 'LayerNorm' in key_str:
                if 'scale' in key_str:
                    set_param(torch_model, f'{base}.attn_ln.weight', value)
                elif 'bias' in key_str:
                    set_param(torch_model, f'{base}.attn_ln.bias', value)
            
            # Handle projection layers
            elif 'Dense_0' in key_str:
                param_name = 'weight' if 'kernel' in key_str else 'bias'
                set_param(torch_model, f'{base}.mlp.proj.{param_name}', value)

    return torch_model

def convert_checkpoint(ckpt_path: str, config: Any) -> torch.nn.Module:
    """Convert a Flax checkpoint to a PyTorch model."""
    # Create PyTorch model
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
    torch_model = TorchTransformerLMHeadModel(torch_config).to(torch.bfloat16)
    

    # Initialize flax model config
    model_config = TransformerConfig(
        dtype=config.dtype,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        emb_dim=config.emb_dim,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        deterministic=True,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, inference_rng = random.split(rng, num=3)

    # Initialize the model and get initial variables
    rng, dropout_rng = jax.random.split(rng)
    input_shape = (config.minibatch_size, config.seq_len)
    net = TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    sample_out, initial_variables = jax.jit(
        net.init_with_output
        )(rng_keys, jnp.ones(input_shape, jnp.int32))
    
    state, lr_scheduler_fn = trainer.get_state(config, net, initial_variables)
    restored_state = checkpoints.restore_checkpoint(ckpt_path, state)
    print("----------Restored model from", ckpt_path, "-----------")
    
    # Convert weights
    convert_flax_to_torch_weights(restored_state, torch_model)
    
    return torch_model

# Example usage
if __name__ == "__main__":
    config = get_config()
    torch_model = convert_checkpoint("/u/pranshu3/llm-reasoning-logic-puzzles/sudoku-code/logs/checkpoint_4192000", config)
    
    # Save PyTorch model
    torch.save(torch_model.state_dict(), "converted_model.pt")
    print("Saved torch checkpoint")