import torch
from torch import Tensor, nn
from typing import Optional, Tuple

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
    
    """
    Forward pass method; attention calculation is done here
    Attention function is as follows: Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
    """
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size: int = query.shape[0]

        # Q, K, and V hold the projected tensors after passing inputs through linear layers
        Q: Tensor = self.fc_q(query)
        K: Tensor = self.fc_k(key)
        V: Tensor = self.fc_v(value)

        # Reshape tensors for multi-head attention calculation
        # Shape becomes [batch_size, h, sequence_length, d_k]
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
        x: Tensor = torch.matmul(self.dropout(attention), V)
        
        # Concatenate heads
        # .view() requires the tensor to be contiguous in memory
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        # Final projection
        x = self.fc_o(x)
        
        return x, attention