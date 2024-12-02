#Pytorch model implementation
import torch
import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import ml_collections
import numpy as np

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
    eval_epochs: int = 1  # Added to match JAX config
    minibatch_size: int = 12

class TorchBilinearDense(torch.nn.Module):
    """PyTorch implementation of BilinearDense layer."""
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, gate_fn: Optional[str] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_fn = gate_fn
        
        # Keep original dimensions but transpose in forward pass
        self.left_weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.right_weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        
        if use_bias:
            self.left_bias = torch.nn.Parameter(torch.empty(out_features))
            self.right_bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('left_bias', None)
            self.register_parameter('right_bias', None)
            
        self.left_ln = torch.nn.LayerNorm(out_features)
        self.right_ln = torch.nn.LayerNorm(out_features)
        self.output_ln = torch.nn.LayerNorm(out_features)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.left_weight, gain=np.sqrt(2.0))
        torch.nn.init.xavier_normal_(self.right_weight, gain=np.sqrt(2.0))
        if self.left_bias is not None:
            torch.nn.init.zeros_(self.left_bias)
            torch.nn.init.zeros_(self.right_bias)
    
    def forward(self, x):
        #print(f"BilinearDense input shape: {x.shape}")
        #print(f"Left weight shape: {self.left_weight.shape}")
        #print(f"Right weight shape: {self.right_weight.shape}")
        
        # The weight matrices need to be transposed for proper matrix multiplication
        left = F.linear(x, self.left_weight.t(), self.left_bias)
        #print(f"After left linear shape: {left.shape}")
        
        right = F.linear(x, self.right_weight.t(), self.right_bias)
        #print(f"After right linear shape: {right.shape}")
        
        left = self.left_ln(left)
        right = self.right_ln(right)
        
        if self.gate_fn == 'gelu':
            left = F.gelu(left)
        elif self.gate_fn == 'relu':
            left = F.relu(left)
        elif self.gate_fn == 'silu':
            left = F.silu(left)
            
        output = left * right
        output = self.output_ln(output)
        #print(f"BilinearDense output shape: {output.shape}")
        return output

class TorchBilinearMLP(torch.nn.Module):
    """PyTorch implementation of BilinearMLP."""
    def __init__(self, config: TorchTransformerConfig):
        super().__init__()
        self.pre_ln = torch.nn.LayerNorm(config.emb_dim)
        self.bilinear = TorchBilinearDense(config.emb_dim, config.mlp_dim)
        self.proj = torch.nn.Linear(config.mlp_dim, config.emb_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        
    def forward(self, x):
        #print(f"BilinearMLP input shape: {x.shape}")
        x = self.pre_ln(x)
        x = self.bilinear(x)
        #print(f"After bilinear shape: {x.shape}")
        x = self.proj(x)
        #print(f"After projection shape: {x.shape}")
        x = self.dropout(x)
        return x
        
class TorchTransformerBlock(torch.nn.Module):
    """PyTorch implementation of TransformerBlock."""
    def __init__(self, config: TorchTransformerConfig):
        super().__init__()
        self.attn_ln = torch.nn.LayerNorm(config.emb_dim)
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout_rate,
            batch_first=True
        )
        self.mlp = TorchBilinearMLP(config)
        
    def forward(self, x, attention_mask=None):
        attn_input = self.attn_ln(x)
        attn_output, _ = self.self_attention(
            attn_input, attn_input, attn_input,
            attn_mask=attention_mask
        )
        x = x + attn_output
        x = x + self.mlp(x)
        return x

class TorchTransformerLMHeadModel(torch.nn.Module):
    def __init__(self, config: TorchTransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = torch.nn.Embedding(
            config.vocab_size, 
            config.emb_dim,
        )
        
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(1, config.seq_len, config.emb_dim)
        )
        
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        
        self.blocks = torch.nn.ModuleList([
            TorchTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = torch.nn.LayerNorm(config.emb_dim)
        self.lm_head = torch.nn.Linear(
            config.emb_dim, 
            config.vocab_size,
            bias=False
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
        torch.nn.init.xavier_uniform_(self.lm_head.weight)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_length, :]
        x = self.dropout(x)
        
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length),
                diagonal=1
            ).bool()
            attention_mask = attention_mask.to(input_ids.device)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits