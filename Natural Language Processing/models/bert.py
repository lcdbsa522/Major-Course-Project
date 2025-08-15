import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class BERTModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=768, dropout=0.1,
                 freeze_pretrained=False, use_pretrained=True,
                 model_name='bert-base-uncased'):
        """
        Simple BERT model for sentiment analysis
        
        Args:
            num_classes: Number of output classes (3 for sentiment)
            hidden_dim: Hidden dimension for classifier
            dropout: Dropout rate
            freeze_pretrained: Whether to freeze pretrained weights
            use_pretrained: Whether to use pretrained BERT
            model_name: Name of pretrained model to use
        """
        super(BERTModel, self).__init__()
        
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Load pretrained BERT
            self.bert = BertModel.from_pretrained(model_name)
            self.config = self.bert.config
            
            if freeze_pretrained:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            # Create BERT from scratch (not recommended)
            config = BertConfig(
                hidden_size=hidden_dim,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout
            )
            self.bert = BertModel(config)
            self.config = config
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim),
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
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            token_type_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits