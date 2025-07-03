import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

# --- 1. Define Tokenizers and Helper Classes ---
# Load spaCy models once to avoid reloading them.
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("Spacy models not found. Please run the following command from your activated environment:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")
    exit()

class ListDataset(Dataset):
    """A simple Dataset class that wraps a list of data."""
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def tokenize_de(text):
    """Tokenizes German text from a string into a list of token strings."""
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """Tokenizes English text from a string into a list of token strings."""
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Helper function to yield tokens for vocabulary building
def yield_tokens(data_list, tokenizer, index):
    """
    Helper function to yield tokens from a list of data.
    `data_list` is a list of raw text pairs.
    `tokenizer` is the appropriate tokenizer function.
    `index` is 0 for source (German) and 1 for target (English).
    """
    for data_sample in data_list:
        yield tokenizer(data_sample[index])

# --- 2. Main Data Loading and Processing Function ---
def get_data_loaders(device, batch_size=128):
    """
    The main function to load data, build vocab, and create DataLoaders.
    """
    # Define special symbols and their indices for the vocabularies
    UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

    # Load Raw Text Data
    train_iter = Multi30k(root='.data', split='train')
    valid_iter = Multi30k(root='.data', split='valid')
    test_iter = Multi30k(root='.data', split='test')

    # Convert the iterators to lists to make them reusable.
    train_data_list = list(train_iter)
    valid_data_list = list(valid_iter)
    test_data_list = list(test_iter)

    # Build Vocabularies
    # Use min_freq=2 to filter out rare tokens and keep vocab size reasonable.
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train_data_list, tokenize_de, 0),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_src.set_default_index(UNK_IDX)

    vocab_trg = build_vocab_from_iterator(
        yield_tokens(train_data_list, tokenize_en, 1),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_trg.set_default_index(UNK_IDX)

    # Collate Function
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            # Tokenize source and target text
            # Convert source text to tensor
            src_tensor = torch.tensor(
                [SOS_IDX] + vocab_src(tokenize_de(src_sample)) + [EOS_IDX],
                dtype=torch.long
            )
            src_batch.append(src_tensor)

            # Convert target text to tensor
            trg_tensor = torch.tensor(
                [SOS_IDX] + vocab_trg(tokenize_en(trg_sample)) + [EOS_IDX],
                dtype=torch.long
            )
            trg_batch.append(trg_tensor)

        # Pad the source and target tensors to the same length so that they can be batched together.
        src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
        
        return src_padded.to(device), trg_padded.to(device)

    # Create DataLoaders
    train_dataloader = DataLoader(ListDataset(train_data_list), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(ListDataset(valid_data_list), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(ListDataset(test_data_list), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Return the vocabularies, DataLoaders, and the padding index.
    return vocab_src, vocab_trg, train_dataloader, valid_dataloader, test_dataloader, PAD_IDX

# --- 3. Test Block ---
if __name__ == '__main__':
    """
    This special block runs only when you execute the script directly from your terminal.
    Usage: python src/data_loader.py
    """
    print("Testing the data loader with the modern torchtext API...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128

    vocab_src, vocab_trg, train_loader, _, _, PAD_IDX = get_data_loaders(
        device=device,
        batch_size=BATCH_SIZE
    )

    print("\nSample Batch")
    for src, trg in train_loader:
        print(f"First batch source shape: {src.shape}")
        print(f"First batch target shape: {trg.shape}")
        print("\nSample source text (from indices):")
        first_src_sample = " ".join([vocab_src.get_itos()[idx] for idx in src[0]])
        print(first_src_sample)
        print("\nSample target text (from indices):")
        first_trg_sample = " ".join([vocab_trg.get_itos()[idx] for idx in trg[0]])
        print(first_trg_sample)
        break

    print("\nData loader test complete!")
