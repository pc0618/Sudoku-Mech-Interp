"""Model Architecture."""

import functools
from typing import Any, Callable

from flax import linen as nn
from flax import struct
from jax import numpy as jnp
from dataclasses import field
from flax import struct

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
        dense_out = nn.Dense(
            features=self.config.emb_dim,
            use_bias=True,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            bias_init=nn.initializers.zeros
        )(x)
        
        # Capture dense output before dropout with module path
        self.sow('intermediates', f'dense_out_{self.name}', dense_out)
        
        # Apply dropout
        x = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=self.config.deterministic
        )(dense_out)
        
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
        mlp_output = BilinearMLP(config=self.config)(mlp_input)
        
        # Store the MLP output in intermediates with the correct key pattern
        self.sow('intermediates', 'block_mlp', mlp_output)
        
        x = x + mlp_output
        
        return x
        
class TransformerLMHeadModel(nn.Module):
    config: Any = None

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.emb_dim = self.config.emb_dim
        self.num_layers = self.config.num_layers

    @nn.compact
    def __call__(self, inputs, training=True):
        batch_size, seq_size = inputs.shape
        causal_mask_x = nn.make_causal_mask(inputs, dtype=self.config.dtype)

        # Embed the input tensor
        embed_with_init = functools.partial(
            nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))
        token_embeddings = embed_with_init(
            num_embeddings=self.config.vocab_size,
            features=self.config.emb_dim,
        )(inputs)

        assert token_embeddings.shape == (batch_size, seq_size, self.config.emb_dim)

        # Position embeddings
        pos_embedding_variable = self.variable(
            "params",
            "position_embeddings",
            jnp.zeros,
            (self.config.seq_len, self.config.emb_dim),
        )
        pos_embeddings = pos_embedding_variable.value[:seq_size, :]
        assert pos_embeddings.shape == token_embeddings.shape[1:]

        x = token_embeddings + pos_embeddings[None, :, :]
        x = nn.Dropout(rate=self.config.dropout_rate,
                    deterministic=self.config.deterministic)(x)

        # Apply Transformer blocks and collect MLP outputs
        for i in range(self.num_layers):
            layer_x = TransformerBlock(config=self.config)(
                    x, causal_mask_x, training=training)
            # Store the output of each layer
            self.sow('intermediates', f'layer_{i}_output', layer_x)
            x = layer_x

        x = nn.LayerNorm()(x)
        self.sow('intermediates', 'pre_logits', x)

        logits = nn.Dense(features=self.config.vocab_size,
                        kernel_init=nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.normal(stddev=1e-6),
                        use_bias=False)(x)

        assert logits.shape == (batch_size, seq_size, self.config.vocab_size)
        return logits