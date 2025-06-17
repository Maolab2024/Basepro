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

# def process_variant(variant: str, WT_sequence: str) -> str:
#     """

#     Args:
#         variant (str): Variant string.
#         WT_sequence (str): Wild-type sequence.

#     Returns:
#         str: Processed variant string.
#     """
#     # Check if variant is WT
#     if variant == 'WT':
#         return variant
    
#     # Extract position and amino acids
#     position = int(variant[:-1])
#     wt_aa = WT_sequence[position - 1]
#     return wt_aa + variant




def create_iteration_dataframes(df_list: List[pd.DataFrame], expected_variants: List[str]) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    """
    Create training and testing dataframes for iterative learning.

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames containing experimental data from each round.
        每轮的实验数据
        expected_variants (List[str]): List of all expected variant names.
        全部的待预测数据

    Returns:
        Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]: 
            - iteration DataFrame: Contains variant and iteration information for training.
            - labels DataFrame: Contains variant, activity, and iteration information for testing.
            Returns (None, None) if duplicates are found.
            返回一个元组，包含两个元素。第一个元素是用于训练的 iteration 数据框，包含变体和迭代信息；第二个元素是用于测试的 labels 数据框，包含变体、活性、迭代信息、二进制活性和缩放后的活性信息
    """
    processed_dfs = []

    # Process each round's data
    for round_num, df in enumerate(df_list, start=1):
        df_copy = df.copy()
        
        # Set iteration for WT in first round, exclude WT from subsequent rounds
        if round_num == 1:
            df_copy.loc[df_copy['updated_variant'] == 'WT', 'iteration'] = 0
        else:
            df_copy = df_copy[df_copy['updated_variant'] != 'WT']
        
        df_copy.loc[df_copy['updated_variant'] != 'WT', 'iteration'] = round_num
        df_copy['iteration'] = df_copy['iteration'].astype(float)
        df_copy.rename(columns={'updated_variant': 'variant'}, inplace=True)
        
        processed_dfs.append(df_copy)

    # Combine all processed dataframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Check for duplicates
    if has_duplicates(combined_df):
        return None, None

    # Create iter_train dataframe
    iteration = combined_df[['variant', 'iteration']]

    # Create iter_test dataframe
    labels = combined_df[['variant', 'activity', 'iteration']]

    # Add a activity_binary and activity_scaled column to labels
    labels['activity_binary'] = labels['activity'].apply(lambda x: 1 if x >= 1 else 0)
    labels['activity_scaled'] = labels['activity'].apply(lambda x: (x - labels['activity'].min()) / (labels['activity'].max() - labels['activity'].min()))

    # Add missing variants to iter_test
    labels = add_missing_variants(labels, expected_variants)

    # Reorder iter_test based on expected variants
    labels = labels.set_index('variant').reindex(expected_variants, fill_value=np.nan).reset_index()
    labels.rename(columns={'index': 'variant'}, inplace=True)
    
    return iteration, labels


def has_duplicates(df: pd.DataFrame) -> bool:
    """
    Check for duplicates in the 'variant' column of the dataframe.

    Args:
        df (pd.DataFrame): DataFrame to check for duplicates.

    Returns:
        bool: True if duplicates are found, False otherwise.
    """
    # Find duplicates in the 'variant' column
    duplicates = df[df.duplicated(subset=['variant'], keep=False)]
    
    # Print duplicates if found
    if not duplicates.empty:
        print("Duplicates found in variant column:")
        print(duplicates)
        print("Exiting.")
        return True
    return False

def add_missing_variants(df: pd.DataFrame, expected_variants: List[str]) -> pd.DataFrame:
    """
    Add missing variants to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to add missing variants to.
        expected_variants (List[str]): List of all expected variant names.

    Returns:
        pd.DataFrame: DataFrame with missing variants added.
    """
    missing_variants = set(expected_variants) - set(df['variant'])
    missing_df = pd.DataFrame({
        'variant': list(missing_variants),
        'activity': np.nan,
        'activity_binary': np.nan,
        'activity_scaled': np.nan,
        'iteration': np.nan
    })
    return pd.concat([df, missing_df], ignore_index=True)