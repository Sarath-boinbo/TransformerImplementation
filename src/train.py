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