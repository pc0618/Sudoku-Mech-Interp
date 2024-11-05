import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    """Global hyperparameters configuration."""
    def __init__(
        self,
        vocab_size=1,
        dtype=torch.float32,
        emb_dim=512,
        num_heads=8,
        num_layers=6,
        qkv_dim=512,
        mlp_dim=2048,
        seq_len=2048,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        deterministic=False
    ):
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qkv_dim = qkv_dim
        self.mlp_dim = mlp_dim
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.qkv_dim // config.num_heads
        self.qkv_dim = config.qkv_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.emb_dim, config.qkv_dim, bias=False)
        self.k_proj = nn.Linear(config.emb_dim, config.qkv_dim, bias=False)
        self.v_proj = nn.Linear(config.emb_dim, config.qkv_dim, bias=False)
        self.out_proj = nn.Linear(config.qkv_dim, config.emb_dim, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout_rate)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask
        if mask is not None:
            mask = mask.view(1, 1, seq_len, seq_len)
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        # Normalize and apply dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.qkv_dim)
        
        # Final projection
        output = self.out_proj(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        
        # MLP layers
        self.mlp_hidden = nn.Linear(config.emb_dim, config.mlp_dim)
        self.mlp_out = nn.Linear(config.mlp_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.mlp_hidden.weight)
        nn.init.normal_(self.mlp_hidden.bias, std=1e-6)
        nn.init.xavier_uniform_(self.mlp_out.weight)
        nn.init.normal_(self.mlp_out.bias, std=1e-6)

    def forward(self, x, mask=None):
        # Attention block (with pre-norm like in JAX implementation)
        normed_x = self.norm1(x)
        attention_output = self.attention(normed_x, mask)
        x = x + attention_output
        
        # MLP block (with pre-norm like in JAX implementation)
        normed_x = self.norm2(x)
        h = F.gelu(self.mlp_hidden(normed_x))
        h = self.dropout(h)
        output = self.mlp_out(h)
        output = self.dropout(output)
        x = x + output
        
        return x

class TransformerLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Positional embedding
        self.register_buffer(
            'pos_embedding',
            torch.zeros(config.seq_len, config.emb_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.emb_dim)
        
        # Output head
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def make_causal_mask(self, input_ids):
        """Generate causal mask similar to JAX implementation."""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create a sequence mask
        mask = torch.ones((seq_length, seq_length), device=device).tril()
        # Make it boolean
        mask = mask.bool()
        
        return mask

    def forward(self, input_ids, training=True):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        pos_embeddings = self.pos_embedding[:seq_length].to(device)
        x = token_embeddings + pos_embeddings[None, :, :]
        
        # Apply embedding dropout
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.make_causal_mask(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Apply language model head
        logits = self.lm_head(x)
        
        return logits