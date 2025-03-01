import torch
import torch.nn as nn
import math

class PromoterGRU(nn.Module):
    """
    GRU-based neural network for promoter prediction from DNA sequences.
    This model uses bidirectional GRU layers with attention mechanism to 
    predict whether a 70nt DNA sequence is a promoter.
    """
    def __init__(self, input_size=4, hidden_size=256, num_layers=3, dropout=0.3, bidirectional=True):
        """
        Initialize the GRU model for promoter prediction.
        
        Args:
            input_size (int): Dimension of input features (4 for one-hot encoded DNA)
            hidden_size (int): Size of the hidden layers
            num_layers (int): Number of GRU layers stacked together
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional GRU
        """
        super().__init__()
        # Enhanced embedding layer to transform one-hot encoding into dense representation
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),  # GELU activation for better performance
            nn.LayerNorm(hidden_size),  # Layer normalization for stable training
            nn.Dropout(dropout)  # Dropout for regularization
        )
        
        # GRU layers to capture sequential patterns in DNA
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Apply dropout between GRU layers if more than 1 layer
            bidirectional=bidirectional,  # Bidirectional to capture patterns in both directions
            batch_first=True  # Expect input shape as (batch_size, seq_length, features)
        )
        
        # Calculate output feature dimension based on bidirectionality
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism to focus on important regions of the sequence
        self.attention = SequenceAttention(gru_output_dim, dropout)
        
        # Classification head with multiple fully-connected layers
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)  # Binary classification output
        )

    def forward(self, x):
        """
        Forward pass of the promoter prediction model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size)
                        containing one-hot encoded DNA sequences
        
        Returns:
            Tensor: Raw logits for promoter prediction (not sigmoid-activated)
        """
        # Transform one-hot encoding to dense representation
        x = self.embedding(x)
        
        # Process sequence through GRU layers
        gru_out, _ = self.gru(x)  # Ignore the hidden state
        
        # Apply attention mechanism to focus on important parts
        context = self.attention(gru_out)
        
        # Final classification
        out = self.fc(context)
        return out  # Return raw logits for use with BCEWithLogitsLoss

class SequenceAttention(nn.Module):
    """
    Attention mechanism for focusing on important parts of the sequence.
    This implementation uses a learnable query to compute attention weights
    over the input sequence.
    """
    def __init__(self, hidden_size, dropout=0.1):
        """
        Initialize the attention mechanism.
        
        Args:
            hidden_size (int): Size of the hidden representations
            dropout (float): Dropout rate for regularization
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Transform input
            nn.Tanh(),  # Non-linearity
            nn.Linear(hidden_size, 1, bias=False)  # Compute attention scores
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Apply attention mechanism to the input sequence.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_size)
                        containing GRU outputs for each position
        
        Returns:
            Tensor: Context vector of shape (batch_size, hidden_size)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # Shape: (batch_size, seq_length, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Apply softmax along sequence dimension
        
        # Apply attention weights to get context vector
        context = torch.bmm(attn_weights.transpose(1, 2), x)  # Shape: (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # Shape: (batch_size, hidden_size)
        
        return self.dropout(context)  # Apply dropout for regularization
