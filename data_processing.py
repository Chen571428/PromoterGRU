from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

def load_dna_data():
    """
    Load the DNA core promoter dataset.
    
    Returns:
        Dataset: Hugging Face dataset containing DNA sequences and promoter labels
    """
    dataset = load_dataset("dnagpt/dna_core_promoter")
    return dataset['train']

def dna_to_onehot(sequences):
    """
    Convert DNA sequences to one-hot encoded tensors.
    
    Args:
        sequences (list): List of DNA sequences, each sequence is a string of nucleotides
    
    Returns:
        Tensor: One-hot encoded tensor of shape (num_sequences, sequence_length, 4)
                where 4 represents the four nucleotides (A, T, C, G)
    """
    vocab = {'A':0, 'T':1, 'C':2, 'G':3}
    onehot = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            if char in vocab:
                onehot[i, j, vocab[char]] = 1.0
            else:  # Handle unknown nucleotides by assigning equal probability
                onehot[i, j] = np.array([0.25, 0.25, 0.25, 0.25])
    return torch.tensor(onehot)

def reverse_complement(sequences):
    """
    Generate reverse complement of DNA sequences for data augmentation.
    The reverse complement is created by reversing the sequence and replacing
    each nucleotide with its complement (A↔T, G↔C).
    
    Args:
        sequences (list): List of DNA sequences
    
    Returns:
        list: List of reverse complemented DNA sequences
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    rev_comp_seqs = []
    
    for seq in sequences:
        rev_comp = ''.join(complement.get(base, 'N') for base in reversed(seq))
        rev_comp_seqs.append(rev_comp)
    
    return rev_comp_seqs

def prepare_data(batch_size=64, augment=True, val_size=0.15, test_size=0.05):
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        batch_size (int): Batch size for training and evaluation
        augment (bool): Whether to augment training data with reverse complements
        val_size (float): Proportion of data to use for validation
        test_size (float): Proportion of data to use for testing
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) containing DataLoader objects
               for training, validation, and testing
    """
    # Load the DNA sequence dataset
    dataset = load_dna_data()
    sequences = dataset['sequence']
    labels = np.array(dataset['label'], dtype=np.float32)
    
    # Data augmentation - add reverse complement sequences (only for training set)
    if augment:
        # First split data to ensure only training data is augmented
        train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=val_size+test_size, stratify=labels, random_state=42
        )
        
        val_seqs, test_seqs, val_labels, test_labels = train_test_split(
            temp_seqs, temp_labels, test_size=test_size/(val_size+test_size), stratify=temp_labels, random_state=42
        )
        
        # Augment training set with reverse complements
        rev_comp_seqs = reverse_complement(train_seqs)
        train_seqs = train_seqs + rev_comp_seqs
        train_labels = np.concatenate([train_labels, train_labels])
    else:
        # No augmentation, just split the data
        train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=val_size+test_size, stratify=labels, random_state=42
        )
        
        val_seqs, test_seqs, val_labels, test_labels = train_test_split(
            temp_seqs, temp_labels, test_size=test_size/(val_size+test_size), stratify=temp_labels, random_state=42
        )
    
    # Convert to one-hot encoding
    train_features = dna_to_onehot(train_seqs)
    val_features = dna_to_onehot(val_seqs)
    test_features = dna_to_onehot(test_seqs)
    
    # Convert to PyTorch tensors
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    # Handle class imbalance with weighted sampling
    class_counts = np.bincount(train_labels.numpy().astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.numpy().astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
