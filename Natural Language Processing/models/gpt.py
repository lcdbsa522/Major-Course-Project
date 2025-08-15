import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config


class GPTModel(nn.Module):
    def __init__(self, vocab_size=50257, max_length=512, num_classes=3,
                 hidden_dim=768, num_layers=12, num_heads=12, dropout=0.1,
                 freeze_pretrained=False, use_pretrained=True):
        """
        Simple GPT model for sentiment analysis
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            num_classes: Number of output classes (3 for sentiment)
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            freeze_pretrained: Whether to freeze pretrained weights
            use_pretrained: Whether to use pretrained GPT-2
        """
        super(GPTModel, self).__init__()
        
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Load pretrained GPT-2
            self.gpt = GPT2Model.from_pretrained('gpt2')
            self.config = self.gpt.config
            
            if freeze_pretrained:
                for param in self.gpt.parameters():
                    param.requires_grad = False
        else:
            # Create GPT from scratch
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=max_length,
                n_embd=hidden_dim,
                n_layer=num_layers,
                n_head=num_heads,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.gpt = GPT2Model(config)
            self.config = config
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.n_embd, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # GPT forward pass
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Use the last token's representation
        if attention_mask is not None:
            # Find the last non-padded token
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=input_ids.device)
            last_hidden = hidden_states[batch_indices, lengths]
        else:
            # If no mask, just use the last token
            last_hidden = hidden_states[:, -1]
        
        # Classification
        logits = self.classifier(last_hidden)
        
        return logits