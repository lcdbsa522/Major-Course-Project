import os
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Import custom modules
from dataset.custom_dataset import SentimentDataset
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.gpt import GPTModel
from models.bert import BERTModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['rnn', 'lstm', 'gpt', 'bert'],
                        help='Type of model to train')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, default='dataset/train_cleaned.csv',
                        help='Path to training data')
    parser.add_argument('--test_data', type=str, default='dataset/test_cleaned.csv',
                        help='Path to test data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Optimizer and Scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'plateau', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for ReduceLROnPlateau')
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (RNN/LSTM)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN/LSTM')
    
    # Transformer specific
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained model (GPT/BERT)')
    parser.add_argument('--freeze_pretrained', action='store_true',
                        help='Freeze pretrained weights')
    
    # Data processing
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size (RNN/LSTM)')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args, vocab_size):
    """Create model based on model type"""
    if args.model_type == 'rnn':
        model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif args.model_type == 'lstm':
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif args.model_type == 'gpt':
        model = GPTModel(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_pretrained=args.use_pretrained,
            freeze_pretrained=args.freeze_pretrained
        )
    elif args.model_type == 'bert':
        model = BERTModel(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_pretrained=args.use_pretrained,
            freeze_pretrained=args.freeze_pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model


def create_optimizer(model, args):
    """Create optimizer based on arguments"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, args, train_loader):
    """Create learning rate scheduler based on arguments"""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # maximize accuracy
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, gradient_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask'].to(device)
            if 'token_type_ids' in batch:
                # BERT
                token_type_ids = batch['token_type_ids'].to(device)
                logits = model(input_ids, attention_mask, token_type_ids)
            else:
                # GPT
                logits = model(input_ids, attention_mask)
        else:
            # RNN/LSTM
            logits = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on data"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask'].to(device)
                if 'token_type_ids' in batch:
                    # BERT
                    token_type_ids = batch['token_type_ids'].to(device)
                    logits = model(input_ids, attention_mask, token_type_ids)
                else:
                    # GPT
                    logits = model(input_ids, attention_mask)
            else:
                # RNN/LSTM
                logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            # Store predictions and labels for F1 score
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Calculate F1 weighted score
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, f1_weighted, all_predictions, all_labels


def save_results(args, history, save_path):
    """Save training results and plot graphs"""
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        return obj
    
    # Convert history to native types
    history_native = convert_to_native(history)
    
    # Save training history
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(history_native, f, indent=2)
    
    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1 Score plot
    if 'test_f1_weighted' in history:
        ax3.plot(history['test_f1_weighted'], label='Test F1 (weighted)', color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Test F1 Score (Weighted)')
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'No F1 Score', ha='center', va='center', transform=ax3.transAxes)
    
    # Gap between train and test accuracy
    gap = [train - test for train, test in zip(history['train_acc'], history['test_acc'])]
    ax4.plot(gap)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gap (%)')
    ax4.set_title('Train-Test Accuracy Gap (Overfitting Indicator)')
    ax4.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Save classification report for best epoch
    if 'best_predictions' in history and 'best_labels' in history:
        class_names = ['Negative', 'Neutral', 'Positive']
        # Convert predictions and labels to native Python lists
        predictions_list = [int(x) for x in history['best_predictions']]
        labels_list = [int(x) for x in history['best_labels']]
        
        report = classification_report(
            labels_list, 
            predictions_list, 
            target_names=class_names,
            output_dict=True
        )
        
        # Convert report to native types
        report_native = convert_to_native(report)
        
        with open(os.path.join(save_path, 'classification_report.json'), 'w') as f:
            json.dump(report_native, f, indent=2)
    
    # Save final results
    final_results = {
        'model_type': args.model_type,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_train_acc': float(history['train_acc'][-1]),
        'final_test_loss': float(history['test_loss'][-1]),
        'final_test_acc': float(history['test_acc'][-1]),
        'final_test_f1_weighted': float(history['test_f1_weighted'][-1]) if 'test_f1_weighted' in history else None,
        'best_test_acc': float(max(history['test_acc'])),
        'best_test_acc_epoch': int(history['test_acc'].index(max(history['test_acc']))) + 1,
        'best_test_f1_weighted': float(max(history['test_f1_weighted'])) if 'test_f1_weighted' in history else None,
        'best_test_f1_epoch': int(history['test_f1_weighted'].index(max(history['test_f1_weighted']))) + 1 if 'test_f1_weighted' in history else None,
        'overfitting_gap': float(history['train_acc'][-1]) - float(history['test_acc'][-1])
    }
    
    with open(os.path.join(save_path, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, args.model_type, timestamp)
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Training {args.model_type.upper()} model")
    print(f"Device: {args.device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Save path: {save_path}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SentimentDataset(
        csv_path=args.train_data,
        model_type=args.model_type,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        test_mode=False
    )
    
    test_dataset = SentimentDataset(
        csv_path=args.test_data,
        model_type=args.model_type,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        test_mode=True
    )
    
    # For RNN/LSTM, share vocabulary
    if args.model_type in ['rnn', 'lstm']:
        test_dataset.set_vocabulary(
            train_dataset.word2idx,
            train_dataset.vocab_size_actual
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    vocab_size = train_dataset.get_vocab_size()
    model = create_model(args, vocab_size)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, train_loader)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1_weighted': [],
        'learning_rates': [],
        'best_predictions': None,
        'best_labels': None
    }
    
    # Early stopping variables
    best_test_acc = 0
    best_test_f1 = 0
    patience_counter = 0
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, args.gradient_clip)
        
        # Evaluate
        test_loss, test_acc, test_f1_weighted, predictions, labels = evaluate(
            model, test_loader, criterion, args.device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1_weighted'].append(test_f1_weighted)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - "
              f"Test F1: {test_f1_weighted:.4f} - "
              f"LR: {current_lr:.2e}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(test_acc)
            else:
                scheduler.step()
        
        # Save checkpoint for best model (based on F1 score)
        if test_f1_weighted > best_test_f1:
            best_test_f1 = test_f1_weighted
            best_test_acc = test_acc
            patience_counter = 0
            
            # Save best predictions and labels for classification report
            history['best_predictions'] = predictions
            history['best_labels'] = labels
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'test_acc': test_acc,
                'test_f1_weighted': test_f1_weighted,
                'args': args
            }, os.path.join(save_path, 'best_model.pth'))
            print(f"  --> New best model saved (Test F1: {test_f1_weighted:.4f}, Acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Best test F1 (weighted): {best_test_f1:.4f}")
    
    # Save results and plots
    save_results(args, history, save_path)
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    main()