# Transformer Implementation: "Attention Is All You Need"

## Project Overview

This project implements the Transformer architecture from scratch, based on the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). The implementation focuses on machine translation using the Multi30k dataset (German to English translation).

**Paper Reference:** [Attention Is All You Need - PDF](https://arxiv.org/pdf/1706.03762.pdf)

## Architecture Deep Dive

*[This section will be completed once the model implementation is finished]*

### Multi-Head Attention
*[Code snippets and mathematical formula explanations will be added here]*

### Positional Encoding
*[Code snippets and mathematical formula explanations will be added here]*

### Encoder-Decoder Architecture
*[Code snippets and mathematical formula explanations will be added here]*

### Feed-Forward Networks
*[Code snippets and mathematical formula explanations will be added here]*

### Layer Normalization
*[Code snippets and mathematical formula explanations will be added here]*

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

## Progress Status

- ✅ **Data Pipeline**: Complete with modern torchtext API
- ✅ **Development Environment**: Properly configured with uv and git
- 🔄 **Model Architecture**: In progress
- ⏳ **Training Loop**: Planned
- ⏳ **Evaluation**: Planned

---

*This README will be updated as the project progresses. Each section will be filled in with detailed code examples and mathematical explanations once the corresponding components are implemented.*
