import torch
import math
from torch import Tensor, nn
from typing import Optional, Tuple

"""
Base values from paper:
- Number of Layers (N): 6 identical layers in both the encoder and decoder stacks.
- Model Dimension (d_model): 512.
- Feed-Forward Dimension (d_ff): 2048.
- Number of Attention Heads (h): 8.
- Key Dimension (d_k): 64 (calculated as d_model / h).
- Value Dimension (d_v): 64 (calculated as d_model / h).
- Dropout Rate (P_drop): 0.1.
- Label Smoothing (ε_ls): 0.1.
"""

"""
First class to emulate the MultiHeadAttention formula that the paper specifies.
Calculates the attention for all heads and outputs the final output tensor and the attention weights.
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, p: float) -> None:
        super().__init__()

        # Make sure model dimension is divisible by heads parameter for proper calculations.
        assert d_model % heads == 0, "Model dimensions must be divisible by head count!"

        # Store dimensions
        self.d_model: int = d_model
        self.heads: int = heads
        self.d_k: int = d_model // heads # Dimension of the keys and queries

        # Create linear layers for the query, key, value, and output
        self.fc_q: nn.Linear = nn.Linear(d_model, d_model)
        self.fc_k: nn.Linear = nn.Linear(d_model, d_model)
        self.fc_v: nn.Linear = nn.Linear(d_model, d_model)
        self.fc_o: nn.Linear = nn.Linear(d_model, d_model)

        # Set dropout value to p to help avoid overfitting
        self.dropout: nn.Dropout = nn.Dropout(p)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass method; attention calculation is done here
        Attention function is as follows: Attention(Q, K, V) = softmax( (QK^T)/sqrt(d_k) ) * V
        """
        batch_size: int = query.shape[0]

        # Q, K, and V hold the projected tensors after passing inputs through linear layers
        Q: Tensor = self.fc_q(query)
        K: Tensor = self.fc_k(key)
        V: Tensor = self.fc_v(value)

        # Reshape tensors for multi-head attention calculation
        # Shape becomes [batch_size, h, sequence_length, d_k] after permute switches h and sequence_length
        Q = Q.view(batch_size, -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # 1. Calculate attention scores
        scores: Tensor = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # 1.5: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 2. Apply softmax
        attention: Tensor = torch.softmax(scores, dim=-1)
        
        # 3. Apply attention to V
        # Shape of x = [batch_size, h, sequence_length, d_k]
        x: Tensor = torch.matmul(self.dropout(attention), V)
        
        # 4. Concatenate heads; .view() requires the tensor to be contiguous in memory
        # Shape of x after permute: [batch_size, sequence_length, h, d_k]
        x = x.permute(0, 2, 1, 3).contiguous()

        # Shape of x after view: [batch_size, sequence_length, d_model]
        # d_model = d_k * h, so the two dimensions get flattened essentially
        x = x.view(batch_size, -1, self.d_model)
        
        # 5. Final projection
        x = self.fc_o(x)
        
        return x, attention

"""
Second class that implements the Position-wise Feed-Forward Network from the paper (FFN(x))
After the tokens have gathered information from each other from the MultiHeadAttention, the FFN provides further computation for each token individually. 
It introduces non-linearity, allowing the model to learn more complex transformations and extract richer 
representations from the information gathered during the attention step.
"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float) -> None:
        super().__init__()

        # First linear layer, expands dimension from d_model to d_ff
        self.fc_1: nn.Linear = nn.Linear(d_model, d_ff)

        # Second linear layer, contracts dimension from d_ff back to d_model
        self.fc_2: nn.Linear = nn.Linear(d_ff, d_model)

        # ReLU activation function
        self.relu: nn.ReLU = nn.ReLU()

        # Set dropout value to p to help avoid overfitting
        self.dropout: nn.Dropout = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass for the FFN.
        Corresponds to the formula: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        """
        # Pass input through the first linear layer, then ReLU, then dropout, 
        # and finally the second linear layer
        return self.fc_2(self.dropout(self.relu(self.fc_1(x))))

"""
Third class that implements the Encoder layer from the paper.
This module contains one block of the encoder stack. It consists of a multi-head 
attention layer and a feed-forward layer, with residual connections and layer normalization.
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, p: float) -> None:
        super().__init__()

        # Self-attention sub-layer
        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, heads, p)
        
        # Position-wise feed-forward sub-layer
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, p)

        # Layer normalization for each sub-layer
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)

        # Dropout for each sub-layer output
        self.dropout1: nn.Dropout = nn.Dropout(p)
        self.dropout2: nn.Dropout = nn.Dropout(p)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Defines the forward pass for the Encoder layer.
        The output of each sub-layer is LayerNorm(x + Sublayer(x)).
        """
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # Pass src as query, key, and value to the self-attention layer
        _src, _ = self.self_attn(src, src, src, src_mask) # NOTE: _src is just temporary variable used so src isn't overwritten
        
        # Sub-Layer 1 Output: Apply dropout, add residual connection, and pass through layer norm
        src = self.norm1(src + self.dropout1(_src))

        # --- Sub-layer 2: Position-wise Feed-Forward ---
        # Pass the output of the first sub-layer to the feed-forward network
        _src = self.feed_forward(src)

        # Sublayer 2 Output: Apply dropout, add residual connection, and pass through layer norm
        src = self.norm2(src + self.dropout2(_src))

        return src

"""
Fourth class that implements the Positional Encoding from the paper.
This injects information about the relative or absolute position of the tokens in the sequence.
The positional encodings have the same dimension as the embeddings so that they can be summed.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, p: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p)

        # Create a positional encoding matrix 'pe' of shape [max_len, d_model]
        # Basically intializes an empty matrix of our specified shape filled with 0
        # This will be overwritten by our sin and cos encoded values
        pe = torch.zeros(max_len, d_model)

        # Creates a column vector of token positions from 0 to max_len-1.
        # This is a 1d tensor that provides the "pos" values needed for the sin
        # and cos functions to calculate the position properly.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 1d tensor of even floats. 2i is the formula's term for every even dimension,
        # which is represented here.
        two_i = torch.arange(0, d_model, 2).float()

        # Calculates the denominator term (10000 ^ (2i/d_model)) from the formula in section 3.5.
        div_term = torch.pow(10000.0, two_i / d_model)
        
        # Calculate the positional encodings using sine for even indices and cosine for odd indices
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        
        # Add a batch dimension so it can be added to the input embeddings
        pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model]

        # Register 'pe' as a buffer. A buffer is part of the module's state 
        # but is not considered a learnable parameter by the optimizer.
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encodings to the input token embeddings.
        """
        # x shape: [batch_size, seq_len, d_model]
        # Add the positional encodings up to the sequence length of the input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

"""
Fifth class that implements the full Encoder from the paper.
This module stacks N EncoderLayer blocks and also handles the initial token 
embedding and positional encoding.
"""
class Encoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, N: int, heads: int, d_ff: int, p: float, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Token embedding layer
        self.tok_embedding: nn.Embedding = nn.Embedding(input_dim, d_model)

        # Positional encoding module
        self.pos_encoding: PositionalEncoding = PositionalEncoding(d_model, p)

        # A stack of N encoder layers, correctly registered using nn.ModuleList
        self.layers: nn.ModuleList = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, p) for _ in range(N)])

        # Store d_model to scale the embeddings
        self.d_model: int = d_model

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Defines the forward pass for the full Encoder.
        """
        # 1. Pass src through the token embedding layer
        # src shape: [batch_size, src_len, d_model]
        src = self.tok_embedding(src)

        # 2. Scale the embeddings by sqrt(d_model)
        # This is a crucial step mentioned in section 3.4 of the paper.
        src = src * (self.d_model**0.5)

        # 3. Add positional encodings
        src = self.pos_encoding(src)

        # 4. Loop through the N encoder layers, passing the output of one as the input to the next
        for layer in self.layers:
            src = layer(src, src_mask)

        return src