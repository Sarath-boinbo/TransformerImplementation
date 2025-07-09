# Transformer Implementation: "Attention Is All You Need"

## Project Overview

This project implements the Transformer architecture from scratch, based on the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). The implementation focuses on machine translation using the Multi30k dataset (German to English translation).

**Paper Reference:** [Attention Is All You Need - PDF](https://arxiv.org/pdf/1706.03762.pdf)

## Project Structure

```
TransformerImplementation/
├── src/
│   ├── data_loader.py      # ✅ Completed
│   ├── model.py            # ✅ Completed
│   ├── train.py            # ⏳ Planned
│   └── utils.py            # ⏳ Planned
├── main.py                 # 🔄 Basic structure
├── pyproject.toml          # ✅ Dependencies configured
├── README.md              # ✅ Updated
└── .gitignore             # ✅ Version control setup
```

## Dependencies

The project uses the following key dependencies (see `pyproject.toml`):
- **PyTorch 2.2.0**: Core deep learning framework
- **torchtext 0.17.0**: Modern text processing utilities
- **spaCy**: Advanced tokenization for German and English
- **NumPy 1.26.4**: Numerical computations
- **uv**: Fast Python package installer and resolver

## Getting Started

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd TransformerImplementation
   uv sync
   ```

3. **Install spaCy models**:
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download de_core_news_sm
   ```

## Architecture Deep Dive

### Multi-Head Attention

Implemented in `src/model.py` (`MultiHeadAttention` class)

**Paper Reference:** Section 3.2.2 "Multi-Head Attention" and Section 3.2.1 "Scaled Dot-Product Attention"

- Computes scaled dot-product attention across multiple heads in parallel. Each head projects the input into query, key, and value spaces, computes attention, and the results are concatenated and projected back to the model dimension.
- Supports masking and dropout on attention weights.

**Paper Formula Implementation:**
- **Scaled Dot-Product Attention**: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ (Section 3.2.1)
- **Multi-Head Attention**: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$ where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ (Section 3.2.2)

**Code snippet:**
```python
class MultiHeadAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        # Linear projections: Q, K, V = query@W^Q, key@W^K, value@W^V
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Scaled dot-product attention: scores = QK^T/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        
        # Apply softmax and multiply by V
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        
        # Final linear projection: output = x@W^O
        x = self.fc_o(x)
        return x, attention
```

### Positional Encoding

Implemented in `src/model.py` (`PositionalEncoding` class)

**Paper Reference:** Section 3.5 "Positional Encoding"

The positional encoding implementation exactly matches the original formula from the "Attention Is All You Need" paper:

**Paper Formula Implementation:**
- **Positional Encoding**: $PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$ and $PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$ (Section 3.5)

**Code Implementation:**
```python
# Paper formula: div_term = 10000^(2i/d_model)
div_term = torch.pow(10000.0, two_i / d_model)

# Paper formula: PE(pos,2i) = sin(pos/10000^(2i/d_model))
pe[:, 0::2] = torch.sin(position / div_term)

# Paper formula: PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
pe[:, 1::2] = torch.cos(position / div_term)
```

This ensures the positional encoding exactly matches the published equations, improving interpretability and correctness.

### Encoder-Decoder Architecture

Implemented in `src/model.py` (`Encoder` and `Decoder` classes)

**Paper Reference:** Section 3.1 "Encoder and Decoder Stacks"

**Encoder:**
- The encoder stacks N identical layers, each with multi-head self-attention and feed-forward sub-layers, with residual connections and layer normalization.
- Token embedding and positional encoding are applied before the stack.
- Embeddings are scaled by $\sqrt{d_{model}}$ as specified in Section 3.4.
- Supports source sequence masking to ignore padding tokens.

**Decoder:**
- The decoder also stacks N identical layers, but with three sub-layers per layer:
  1. Masked multi-head self-attention (prevents attending to future tokens)
  2. Multi-head encoder-decoder attention (attends to encoder output)
  3. Position-wise feed-forward network
- Each sub-layer has residual connections and layer normalization.
- Supports both target sequence masking (for future tokens) and source sequence masking.

**Paper Formula Implementation:**
- **Embedding Scaling**: $\text{Embedding} \times \sqrt{d_{model}}$ (Section 3.4)
- **Sub-layer Output**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ (Section 3.1)

**Code snippet:**
```python
class Encoder(nn.Module):
    def forward(self, src, src_mask):
        # Paper formula: Embedding * √d_model (Section 3.4)
        src = self.tok_embedding(src)
        src = src * (self.d_model**0.5)
        src = self.pos_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class Decoder(nn.Module):
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Paper formula: Embedding * √d_model (Section 3.4)
        trg = self.tok_embedding(trg)
        trg = trg * (self.d_model**0.5)
        trg = self.pos_encoding(trg)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        return trg, attention
```

### Feed-Forward Networks

Implemented in `src/model.py` (`PositionwiseFeedForward` class)

**Paper Reference:** Section 3.3 "Position-wise Feed-Forward Networks"

- Applies two linear transformations with a ReLU activation in between, independently to each position.
- Expands dimension from `d_model` to `d_ff`, then projects back.
- Dropout after activation.

**Paper Formula Implementation:**
- **Position-wise Feed-Forward**: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ (Section 3.3)

**Code snippet:**
```python
class PositionwiseFeedForward(nn.Module):
    def forward(self, x):
        # Paper formula: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        return self.fc_2(self.dropout(self.relu(self.fc_1(x))))
```

### Layer Normalization

Implemented in `src/model.py` (used in `EncoderLayer` and `DecoderLayer`)

**Paper Reference:** Section 3.1 "Encoder and Decoder Stacks"

- Each sub-layer (attention, feed-forward) is followed by a residual connection and layer normalization.
- Two layer norms per encoder layer, three per decoder layer.
- Dropout applied to sub-layer outputs before addition.

**Paper Formula Implementation:**
- **Sub-layer Output**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ (Section 3.1)

**Code snippet:**
```python
class EncoderLayer(nn.Module):
    def forward(self, src, src_mask):
        # Paper formula: LayerNorm(x + Sublayer(x)) (Section 3.1)
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(_src))
        _src = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(_src))
        return src

class DecoderLayer(nn.Module):
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Paper formula: LayerNorm(x + Sublayer(x)) (Section 3.1)
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout1(_trg))
        _trg, attention = self.encoder_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout2(_trg))
        _trg = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout3(_trg))
        return trg, attention
```

### Complete Transformer Model

Implemented in `src/model.py` (`Transformer` class)

**Paper Reference:** Section 3 "Model Architecture" (Complete model)

The complete Transformer model combines the encoder and decoder with proper masking mechanisms:

**Key Features:**
- **Source Masking**: Prevents attention to padding tokens in the source sequence
- **Target Masking**: Combines padding mask with causal mask to prevent attending to future tokens
- **Weight Sharing**: Decoder embedding weights are shared with the final linear layer (Section 3.4)
- **Attention Visualization**: Returns attention weights for potential visualization

**Paper Formula Implementation:**
- **Weight Sharing**: "We also use the learned embeddings to convert the output probabilities to next-token probabilities" (Section 3.4)

**Masking Implementation:**
```python
def create_src_mask(self, src):
    # Paper: Masks padding tokens in source sequence
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_trg_mask(self, trg):
    # Paper: Combines padding mask with causal mask for autoregressive generation
    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask
```

**Forward Pass:**
```python
def forward(self, src, trg):
    # Paper: Complete transformer forward pass
    src_mask = self.create_src_mask(src)
    trg_mask = self.create_trg_mask(trg)
    enc_src = self.encoder(src, src_mask)
    output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
    output = self.fc_out(output)
    return output, attention
```

## Data Preprocessing Details

The data pipeline is implemented in `src/data_loader.py` and follows the modern torchtext API approach:

### Tokenization Strategy
- **spaCy Integration**: Uses `en_core_web_sm` for English and `de_core_news_sm` for German tokenization
- **Special Tokens**: Implements `<unk>`, `<pad>`, `<sos>`, `<eos>` with indices 0, 1, 2, 3 respectively
- **Vocabulary Building**: Uses `build_vocab_from_iterator` with `min_freq=2` to filter rare tokens

### Data Pipeline Components

```python
# Key components from data_loader.py:

# 1. Tokenization Functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# 2. Vocabulary Building
vocab_src = build_vocab_from_iterator(
    yield_tokens(train_data_list, tokenize_de, 0),
    min_freq=2,
    specials=special_symbols,
    special_first=True
)

# 3. Custom Collate Function
def collate_fn(batch):
    # Handles padding and tensor conversion
    # Adds SOS/EOS tokens to sequences
    # Returns padded tensors ready for batching
```

### Batching Strategy
- **Batch Size**: Configurable (default: 128)
- **Padding**: Uses `pad_sequence` with `batch_first=True`
- **Device Handling**: Automatically moves tensors to specified device (CPU/GPU)

### Dataset Structure
- **Source**: German text from Multi30k dataset
- **Target**: English text from Multi30k dataset
- **Format**: Raw text pairs → Tokenized → Padded tensors

## Model Architecture Details

### Implementation Insights

**Multi-Head Attention Implementation:**
- **Head Division**: Model dimension is divided equally among heads (`d_k = d_model // heads`)
- **Tensor Reshaping**: Uses `permute(0, 2, 1, 3)` to switch head and sequence dimensions
- **Scaling**: Attention scores are scaled by `1/sqrt(d_k)` as per the paper
- **Masking**: Supports optional masking with `masked_fill(mask == 0, -1e9)`

**Positional Encoding:**
- **Sinusoidal Functions**: Uses both sine and cosine functions for even/odd dimensions
- **Frequency Calculation**: Denominator calculated as `10000^(2i/d_model)` for dimension `i`
- **Buffer Registration**: Positional encodings stored as non-trainable buffers

**Residual Connections:**
- **Pre-Norm Architecture**: Layer normalization applied before residual connections
- **Dropout Integration**: Dropout applied to sub-layer outputs before addition
- **Gradient Flow**: Residual connections help maintain gradient flow through deep networks

### Model Components Summary

| Component | Class | Paper Section | Key Features |
|-----------|-------|---------------|--------------|
| Multi-Head Attention | `MultiHeadAttention` | 3.2.1-3.2.2 | 8 heads, scaled dot-product, masking support |
| Position-wise FFN | `PositionwiseFeedForward` | 3.3 | Two linear layers with ReLU, dropout |
| Encoder Layer | `EncoderLayer` | 3.1 | Self-attention + FFN, residual connections |
| Decoder Layer | `DecoderLayer` | 3.1 | Masked self-attention + encoder-attention + FFN |
| Positional Encoding | `PositionalEncoding` | 3.5 | Sinusoidal encoding, max_len=5000 |
| Encoder | `Encoder` | 3.1 | N stacked layers, embedding + positional encoding |
| Decoder | `Decoder` | 3.1 | N stacked layers, embedding + positional encoding |
| Transformer | `Transformer` | 3.1-3.5 | Complete model with masking and weight sharing |

### Paper Hyperparameters Implementation

**Paper Reference:** Section 3 "Model Architecture" and Table 1 "Model Variations"

The implementation follows the paper's base model hyperparameters:

| Hyperparameter | Paper Value | Implementation |
|----------------|-------------|----------------|
| d_model | 512 | `d_model` parameter |
| d_ff | 2048 | `d_ff` parameter |
| h (heads) | 8 | `heads` parameter |
| N (layers) | 6 | `N` parameter |
| Dropout | 0.1 | `p` parameter |
| max_len | 5000 | `max_len` parameter in PositionalEncoding |

**Key Paper Formulas Implemented:**
- **Attention**: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ (Section 3.2.1)
- **Multi-Head**: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$ (Section 3.2.2)
- **FFN**: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ (Section 3.3)
- **Positional Encoding**: $PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$ (Section 3.5)
- **Sub-layer Output**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ (Section 3.1)
- **Embedding Scaling**: $\text{Embedding} \times \sqrt{d_{model}}$ (Section 3.1)

## Training Details & Results

*[This section will be completed once training is implemented]*

### Hyperparameters
*[To be filled in]*

### Optimizer Configuration
*[To be filled in]*

### Loss Function
*[To be filled in]*

### Learning Rate Schedule
*[To be filled in]*

### Training Results
*[To be filled in]*

## Challenges & Insights

### Development Environment Setup

**Major Challenge: Package Compatibility Issues**

During the initial setup, I encountered significant challenges with package compatibility that taught me valuable lessons about modern Python development:

#### The Problem
- **torchtext Compatibility**: The modern torchtext API (0.17.0) has completely different interfaces compared to older versions
- **spaCy Integration**: Required specific model downloads and proper error handling
- **Dependency Management**: Complex interdependencies between PyTorch, torchtext, and spaCy

#### The Solution
```python
# Robust error handling for spaCy models
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("Spacy models not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")
    exit()
```

#### Key Learnings

1. **Version Pinning is Critical**: 
   - Learned to pin exact versions in `pyproject.toml`
   - `torch == 2.2.0`, `torchtext == 0.17.0` and `numpy == 1.26.4` to ensure compatibility

2. **Modern Python Development Tools**:
   - **uv**: Discovered the power of `uv` for fast dependency resolution
   - **Git Integration**: Learned the importance of proper `.gitignore` and version control
   - **Virtual Environments**: Proper isolation prevents dependency conflicts

3. **Debugging Strategies**:
   - **Incremental Testing**: Test each component separately before integration
   - **Error Messages**: Learned to read and interpret complex error messages
   - **Documentation**: Always check the latest API documentation for breaking changes

### Data Pipeline Insights

**Challenge: Understanding Modern torchtext API**

The transition from older torchtext APIs to the modern version required significant research:

```python
# Modern approach vs old approach
# OLD: torchtext.data.Field (deprecated)
# NEW: build_vocab_from_iterator + custom collate functions
```

**Key Insights:**
- Modern torchtext is more flexible but requires more manual implementation
- Custom collate functions provide better control over batching
- Proper vocabulary building with special tokens is crucial for transformer training

### Model Implementation Challenges

**Challenge 1: Multi-Head Attention Tensor Shapes**
- **Problem**: Complex tensor reshaping for multi-head attention with proper broadcasting
- **Solution**: Used `permute(0, 2, 1, 3)` to switch head and sequence dimensions
- **Learning**: Understanding tensor broadcasting and memory layout is crucial for attention mechanisms

**Challenge 2: Attention Masking Implementation**
- **Problem**: Properly implementing causal masking for decoder self-attention
- **Solution**: Combined padding mask with causal mask using `torch.tril()` and bitwise operations
- **Learning**: Masking is essential for preventing information leakage in autoregressive models

**Challenge 3: Residual Connections and Layer Normalization**
- **Problem**: Ensuring proper gradient flow through deep networks
- **Solution**: Applied layer normalization before residual connections (pre-norm architecture)
- **Learning**: The order of operations significantly affects training stability

### Future Challenges to Address

1. **Training Implementation**:
   - Custom learning rate schedule with warmup
   - Gradient clipping implementation
   - Label smoothing for loss function

2. **Performance Optimization**:
   - Memory management for large batches
   - GPU utilization optimization
   - Training time estimation

3. **Model Evaluation**:
   - BLEU score calculation
   - Attention visualization tools
   - Model interpretability analysis