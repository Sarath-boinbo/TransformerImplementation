# Transformer Implementation: "Attention Is All You Need"

## Project Overview

This project implements the Transformer architecture from scratch, based on the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). The implementation focuses on machine translation using the Multi30k dataset (German to English translation).

**Paper Reference:** [Attention Is All You Need - PDF](https://arxiv.org/pdf/1706.03762.pdf)

## Project Structure

```
TransformerImplementation/
├── src/
│   ├── data_loader.py      # ✅ Completed
│   ├── model.py            # 🔄 In Progress
│   ├── training.py         # ⏳ Planned
│   └── utils.py            # ⏳ Planned
├── main.py                 # 🔄 Basic structure
├── pyproject.toml          # ✅ Dependencies configured
├── README.md              # 🔄 This file
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

- Computes scaled dot-product attention across multiple heads in parallel. Each head projects the input into query, key, and value spaces, computes attention, and the results are concatenated and projected back to the model dimension.
- Supports masking and dropout on attention weights.

**Code snippet:**
```python
class MultiHeadAttention(nn.Module):
    ...
    def forward(self, query, key, value, mask=None):
        # Linear projections, reshape for heads
        # Scaled dot-product attention
        # Masking, softmax, dropout
        # Concatenate heads, final linear projection
        return x, attention
```

### Positional Encoding

The positional encoding implementation in `src/model.py` matches the original formula from the "Attention Is All You Need" paper:
- The denominator is calculated as $10000^{2i/d_{model}}$.
- The sine and cosine functions use `position / div_term` as in the paper.

**Code Snippet:**
```python
# div_term calculation and usage
div_term = torch.pow(10000.0, two_i / d_model)
pe[:, 0::2] = torch.sin(position / div_term)
pe[:, 1::2] = torch.cos(position / div_term)
```

This ensures the positional encoding exactly matches the published equations, improving interpretability and correctness.

### Encoder-Decoder Architecture

Implemented in `src/model.py` (`Encoder` class, encoder only so far)

- The encoder stacks N identical layers, each with multi-head self-attention and feed-forward sub-layers, with residual connections and layer normalization.
- Token embedding and positional encoding are applied before the stack.
- Embeddings are scaled by $\sqrt{d_{model}}$.

**Code snippet:**
```python
class Encoder(nn.Module):
    ...
    def forward(self, src, src_mask):
        src = self.tok_embedding(src)
        src = src * (self.d_model**0.5)
        src = self.pos_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
```

### Feed-Forward Networks

Implemented in `src/model.py` (`PositionwiseFeedForward` class)

- Applies two linear transformations with a ReLU activation in between, independently to each position.
- Expands dimension from `d_model` to `d_ff`, then projects back.
- Dropout after activation.

**Code snippet:**
```python
class PositionwiseFeedForward(nn.Module):
    ...
    def forward(self, x):
        return self.fc_2(self.dropout(self.relu(self.fc_1(x))))
```

### Layer Normalization

Implemented in `src/model.py` (used in `EncoderLayer`)

- Each sub-layer (attention, feed-forward) is followed by a residual connection and layer normalization.
- Two layer norms per encoder layer. Dropout applied to sub-layer outputs before addition.

**Code snippet:**
```python
class EncoderLayer(nn.Module):
    ...
    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(_src))
        _src = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(_src))
        return src
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

### Future Challenges to Address

*[These will be documented as the project progresses]*

1. **Model Implementation Challenges**:
   - Multi-head attention tensor shape debugging
   - Proper implementation of attention masks
   - Residual connections and layer normalization

2. **Training Challenges**:
   - Custom learning rate schedule with warmup
   - Gradient clipping implementation
   - Label smoothing for loss function

3. **Performance Optimization**:
   - Memory management for large batches
   - GPU utilization optimization
   - Training time estimation