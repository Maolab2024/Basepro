import os
from typing import List, Dict, Any, Tuple, Union
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import re

def load_embeddings(base_path: str, embeddings_file_name: str,device: str) -> torch.Tensor:
    """
    Load experimental embeddings from a PyTorch (.pt) file that may not contain sequence names.

    Args:
        base_path (str): Base path to the data directory.
        embeddings_file_name (str): Name of the embeddings file.
        device (str): Target device ('cuda' or 'cpu'). Default 'cpu'

    Returns:
        torch.Tensor: Loaded embeddings tensor
    """
    file_path = os.path.join(base_path, embeddings_file_name)
    
    try:
        # Load .pt file (PyTorch tensor)
        embeddings = torch.load(file_path, map_location=device)
        print(f"Loaded embeddings from {file_path} with shape {embeddings.shape}")
        return embeddings
    
    except Exception as e:
        print(f"Failed to load file: {e}")
        return torch.tensor([])  # Return empty tensor

def load_round_data(round_base_path: str, round_file_names, protein_name: str) -> List[pd.DataFrame]:
    """
    Load round data from CSV files in round order (round0, round1, ...).
    
    Parameters:
    protein_name (str): Name of the protein 
    round_base_path (str): Base path for round data
    
    Returns:
    list: Combined DataFrame from all CSV files in the round, in round order
    """
    all_files = []
    # Collect all matching files
    for file_name in round_file_names:
        if file_name.startswith(protein_name) and file_name.endswith('.csv'):
            file_path = os.path.join(round_base_path, file_name)
            all_files.append(file_path)
    
    # Key step for sorting by round number
    def extract_round_number(file_path):
        """Extract round number from filename"""
        # Use regex to match roundX pattern
        match = re.search(r'round_(\d+)', file_path)
        if match:
            return int(match.group(1))
        # If no round number found, return -1 to place first
        return -1
    
    # Sort files by round number
    sorted_files = sorted(all_files, key=extract_round_number, reverse=False)
    
    # Load data in sorted order
    all_round_data = []
    for file_path in sorted_files:
        df = pd.read_csv(file_path)
        all_round_data.append(df)
        print(f"Loaded: {os.path.basename(file_path)} (Round {extract_round_number(file_path)})")
    
    return all_round_data



