from torch import nn

"""
First class to emulate the MultiHeadAttention formula that the paper specifies.
Calculates the attention for all heads and outputs the final output tensor and the attention weights.
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, p) -> None:

        # Make sure model dimension is divisible by heads parameter for proper calculations.
        assert d_model % heads == 0, "Model dimensions must be divisible by head count!"
        d_k = d_model / heads # Dimension of the keys and queries within a single attention head.

        # Linear layers for the query, key, value, and output respectively. "fc" = fully connected.
        # Each variable is read as "the fully connected layer for the ___"
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # Set dropout value to p to help avoid overfitting
        self.dropout = nn.Dropout(p)
    
    