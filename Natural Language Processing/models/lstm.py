import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2,
                 num_classes=3, dropout=0.3, bidirectional=True):
        """
        Simple LSTM model for sentiment analysis
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (3 for sentiment)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layers
        lstm_output_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)  # padding token
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.)
        
        # Initialize classifier
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids):
        """
        Forward pass
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)
        
        # LSTM forward
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * num_directions)
        
        # Use the last non-padded output
        # Find actual lengths
        lengths = (input_ids != 0).sum(dim=1).long()  # (batch_size,)
        lengths = lengths.clamp(min=1) - 1  # Adjust for 0-indexing
        
        # Gather the last outputs
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        last_outputs = lstm_out[batch_indices, lengths]  # (batch_size, hidden_dim * num_directions)
        
        # Classification
        logits = self.classifier(last_outputs)
        
        return logits