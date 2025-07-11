import torch
from torch import Tensor, nn
from typing import Any, Optional, Tuple

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

"""
Sixth class that implements the Decoder layer from the paper.
This is similar to the EncoderLayer, but has an additional attention layer 
to look at the Encoder's output.
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, p: float) -> None:
        super().__init__()

        # Masked self-attention sub-layer for the target sequence
        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, heads, p)

        # Encoder-decoder attention sub-layer
        self.encoder_attn: MultiHeadAttention = MultiHeadAttention(d_model, heads, p)

        # Position-wise feed-forward sub-layer
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, p)

        # Layer normalization for each of the three sub-layers
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm3: nn.LayerNorm = nn.LayerNorm(d_model)

        # Dropout for each of the three sub-layers
        self.dropout1: nn.Dropout = nn.Dropout(p)
        self.dropout2: nn.Dropout = nn.Dropout(p)
        self.dropout3: nn.Dropout = nn.Dropout(p)

    def forward(self, trg: Tensor, enc_src: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tuple[Tensor, Any]:
        """
        Defines the forward pass for the Decoder layer.
        It has three sub-layers, each followed by a residual connection and layer norm.
        """
        # --- Sub-layer 1: Masked Multi-Head Self-Attention ---
        # The target mask (trg_mask) is used here to prevent attending to future tokens.
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout1(_trg))

        # --- Sub-layer 2: Encoder-Decoder Attention ---
        # The query comes from the previous sub-layer's output (trg).
        # The key and value come from the encoder's output (enc_src).
        # The source mask (src_mask) is used here.
        _trg, attention = self.encoder_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout2(_trg))

        # --- Sub-layer 3: Position-wise Feed-Forward ---
        _trg = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout3(_trg))

        # The attention weights from the encoder-decoder attention are also returned
        # for potential visualization later.
        return trg, attention

"""
Seventh class that implements the full Decoder from the paper.
This module stacks N DecoderLayer blocks and also handles the initial token
embedding and positional encoding for the target sequence.
"""
class Decoder(nn.Module):
    def __init__(self, output_dim: int, d_model: int, N: int, heads: int, d_ff: int, p: float, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Token embedding layer for the target vocabulary
        self.tok_embedding: nn.Embedding = nn.Embedding(output_dim, d_model)

        # Positional encoding module
        self.pos_encoding: PositionalEncoding = PositionalEncoding(d_model, p)

        # A stack of N decoder layers, correctly registered using nn.ModuleList
        self.layers: nn.ModuleList = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, p) for _ in range(N)])

        # Store d_model to scale the embeddings
        self.d_model: int = d_model

    def forward(self, trg: Tensor, enc_src: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Defines the forward pass for the full Decoder.
        """
        # 1. Pass the target sequence through the token embedding layer
        # trg shape: [batch_size, trg_len, d_model]
        trg = self.tok_embedding(trg)

        # 2. Scale the embeddings by sqrt(d_model)
        trg = trg * (self.d_model**0.5)

        # 3. Add positional encodings
        trg = self.pos_encoding(trg)

        # 4. Loop through the N decoder layers
        for layer in self.layers:
            # The DecoderLayer returns the processed target tensor and the attention weights
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # Return the final output tensor and the attention weights from the last layer
        return trg, attention

"""
Eighth and final class that assembles the full Transformer model.
This top-level module brings the Encoder and Decoder together and handles the 
mask creation logic.
"""
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_idx: int, trg_pad_idx: int, device: torch.device) -> None:
        super().__init__()
        
        # Store the instantiated encoder and decoder modules
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

        # Store padding indices and the device, needed for mask creation
        self.src_pad_idx: int = src_pad_idx
        self.trg_pad_idx: int = trg_pad_idx
        self.device: torch.device = device
        
        # Final linear layer to get prediction scores over the target vocabulary
        # The output dimension is the size of the target vocabulary
        output_dim = self.decoder.tok_embedding.num_embeddings
        d_model = self.encoder.d_model
        self.fc_out: nn.Linear = nn.Linear(d_model, output_dim)

        # The paper mentions sharing weights between the decoder's embedding layer
        # and the final linear layer. This is a common practice to reduce parameters.
        self.fc_out.weight = self.decoder.tok_embedding.weight
        
    def create_src_mask(self, src: Tensor) -> Tensor:
        """
        Creates a mask for the source sequence to ignore padding tokens.
        """
        # src shape: [batch_size, src_len]
        # Creates a boolean mask where pad tokens are False.
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # Final shape: [batch_size, 1, 1, src_len] for broadcasting.
        return src_mask
    
    def create_trg_mask(self, trg: Tensor) -> Tensor:
        """
        Creates a combined mask for the target sequence. It masks both padding tokens
        and future tokens.
        """
        # trg shape: [batch_size, trg_len]
        trg_len = trg.shape[1]
        
        # 1. Create a padding mask for the target sequence
        # Shape: [batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Create a "subsequent" or "look-ahead" mask
        # This prevents a position from attending to future positions.
        # Shape: [trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        # 3. Combine the two masks with a bitwise AND
        # The final mask is True only for non-padding tokens in valid positions.
        # Broadcasting handles the shape mismatch correctly.
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask
        
    def forward(self, src: Tensor, trg: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Defines the forward pass for the entire Transformer model.
        """
        # 1. Create masks for the source and target sequences
        src_mask: Tensor = self.create_src_mask(src)
        trg_mask: Tensor = self.create_trg_mask(trg)
        
        # 2. Pass the source sequence and its mask to the encoder
        enc_src: Tensor = self.encoder(src, src_mask)
        
        # 3. Pass the target sequence, encoder output, and masks to the decoder
        output: Tensor
        attention: Tensor
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        # 4. Pass the decoder's output through the final linear layer
        output = self.fc_out(output)
        
        # Return the final prediction scores and the attention weights
        return output, attention