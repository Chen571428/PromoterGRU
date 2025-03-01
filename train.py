import torch
from torch import nn, optim
import numpy as np
from data_processing import prepare_data
from model import PromoterGRU
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
from torch.amp import autocast, GradScaler

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader containing the evaluation dataset
        criterion: Loss function to compute the loss
        device: Device to run the evaluation on (CPU or GPU)
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(device_type=device.type):  # Use mixed precision for efficiency
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            
            # Apply sigmoid to get probabilities (0-1 range)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    metrics = {
        'loss': val_loss / len(data_loader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics

def train_model():
    """
    Train the GRU model for promoter prediction, with validation and testing.
    The function handles:
    - Data loading and preparation
    - Model initialization and training
    - Evaluation on validation set
    - Early stopping based on validation performance
    - Final evaluation on test set
    - Plotting and saving training history
    
    Returns:
        None
    """
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 200
    weight_decay = 0.01
    early_stop_patience = 15
    grad_clip = 1.0
    
    # Select appropriate device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data loaders for training, validation, and testing
    train_loader, val_loader, test_loader = prepare_data(batch_size)
    print("Data prepared!")
    
    # Initialize model and move to the selected device
    model = PromoterGRU().to(device)
    
    # Use binary cross-entropy with logits for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Configure optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler for adaptive learning rate
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,  # Percentage of iterations for the learning rate to increase
        div_factor=10.0,  # Initial learning rate = max_lr/div_factor
        final_div_factor=1000.0  # Final learning rate = max_lr/(div_factor*final_div_factor)
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize dictionary to track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Variables for tracking best model and early stopping
    best_val_auc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    
    start_time = time.time()
    print("Train started!")

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            
            # Use mixed precision for faster training on compatible GPUs
            with autocast(device_type=device.type):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update parameters with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Record training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print training information
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f}')
        print(f'Val Recall: {val_metrics["recall"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        print(f'Val AUC: {val_metrics["auc"]:.4f}')
        
        # Save best model based on validation AUC
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model!')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping if no improvement for specified number of epochs
        if no_improve_epochs >= early_stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    training_time = time.time() - start_time
    print(f'Training complete in {training_time:.2f}s. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}')
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curve
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score curve
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot AUC curve
    plt.subplot(2, 2, 4)
    plt.plot(history['val_auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Print test set evaluation results
    print("\nTest Set Evaluation:")
    print(f'Test Loss: {test_metrics["loss"]:.4f}')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test Precision: {test_metrics["precision"]:.4f}')
    print(f'Test Recall: {test_metrics["recall"]:.4f}')
    print(f'Test F1: {test_metrics["f1"]:.4f}')
    print(f'Test AUC: {test_metrics["auc"]:.4f}')

if __name__ == '__main__':
    train_model()