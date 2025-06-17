import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import List, Dict, Any, Optional, Tuple
import torch
import os

from src.data import load_embeddings, load_round_data,create_iteration_dataframes

def run_directed_evolution(
    protein_name : str,
    round_name : str,
    embeddings_base_path : str,
    embeddings_file_name : str,
    round_base_path : str,
    round_file_names : List[str],
    number_of_variants : int = 90, 
    output_dir : str = "data/output",
    regression_model : str = 'xgboost',
    all_variants: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 

    """
    Perform one round of directed evolution for a protein.

    Args:
    protein_name (str): Name of the protein.
    round_name (str): Name of the current round (e.g., 'Round1').
    embeddings_base_path (str): Base path for embeddings file.
    embeddings_file_name (str): Name of the embeddings file.
    round_base_path (str): Base path for round data files.
    round_file_names (list): List of round file names.
    number_of_variants (int): Number of top variants to display.
    output_dir (str): Directory to save output files.
    regression_model (str): Type of regression model to use (default is 'xgboost').
    all_variants (pd.DataFrame, optional): DataFrame containing all variants and their fitness values are nan. If None, will be loaded from round data.
    Returns:
    tuple: (df_next_round, df_pre_all_sorted)
    """
    
    print(f"Processing {protein_name} - {round_name}")

    # Load embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    embeddings = load_embeddings(embeddings_base_path, embeddings_file_name, device)
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # Load round data
    all_round_data = load_round_data(round_base_path, round_file_names, protein_name)

    
    
    # Perform baes model analysis
    df_next_round, df_pre_all_sorted = base_model(
        embeddings = embeddings,
        all_round_data= all_round_data,
        all_variants=all_variants,
        number_of_variants=number_of_variants,
        regression_model = regression_model,
        device= device,
        experimental = True

    )
    

    # Print results
    print(f"\nTop {number_of_variants} variants predicted by the modelf or next round: {len(df_next_round)}")
    print(df_next_round)
  
    
    # Save results if an output_dir is provided
    if output_dir is not None:
        output_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(output_dir, exist_ok=True)


        # save the next round data to a CSV file for next training
        # 注意正式实验时屏蔽这里，想要在实验之后，手动添加csv文件中的fitness作为训练集！！！！！！！！！！！！！！！！！！！
        filepath = os.path.join(round_base_path, f"{protein_name}_{round_name}.csv")
        df_next_round.to_csv(filepath, index=False)


        # save all variants with predictions to output directory
        df_next_round.to_csv(os.path.join(output_dir, 'next_round_variants.csv'))
        df_pre_all_sorted.to_csv(os.path.join(output_dir, 'df_pre_all_sorted.csv'))
        print(f"\nData saved to {output_dir}")



    
    return df_next_round, df_pre_all_sorted




# Active learning function for one iteration
def base_model(
            embeddings: torch.Tensor,
            all_round_data: List[pd.DataFrame] = None,
            all_variants: Optional[pd.DataFrame] = None,
            number_of_variants: int = 90,
            regression_model: str = 'xgboost',
            device: str = 'cpu',
            experimental: bool = True):
    """
    Perform 

    Args:
    experimental (bool): If True, returns  data for experimental purposes,if False eturns  data for test purposes,.
    Returns:
    tuple: (this_round_variants, df_test, df_sorted_all)
    """
    
    all_X = []
    all_y = []
    list_indices = []

    for df in all_round_data:
        X_round = embeddings[df['indices'].values]
        y_round = torch.tensor(df['fitness'].values,dtype=torch.float32)
        round_indices = df['indices']

        all_X.append(X_round)
        all_y.append(y_round)
        list_indices.append(round_indices)
    X_train = torch.cat(all_X, dim=0)
    y_train = torch.cat(all_y, dim=0) 

    train_indices = pd.concat(list_indices, ignore_index=True)
    test_indices = np.array([i for i in range(embeddings.shape[0]) if i not in train_indices])

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # fit
    if regression_model == 'ridge':
        model = linear_model.RidgeCV()
    
    elif regression_model == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100, 
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            tree_method='hist',
            device = 'cuda'
        )
        


    model.fit(X_train, y_train)

    all_predictions = model.predict(embeddings)

    # # 评估结果 (在训练集和测试集上)(需要完善)
    # train_predictions = all_predictions[train_indices]
    # test_predictions = all_predictions[test_indices]


    
    df_pre_all= pd.DataFrame({
        'variant': all_variants['variant'],
        'fitness': all_predictions
    })
    df_pre_all_sorted = df_pre_all.sort_values(by='fitness', ascending=False)
    filtered_df = df_pre_all_sorted[~df_pre_all_sorted.index.isin(train_indices)]

    # 取前 number_of_variants 个变异体作为下一轮
    selected_variants = filtered_df.head(number_of_variants)

    df_next_round = selected_variants[['variant', 'fitness']].copy()
    df_next_round['indices'] = selected_variants.index  # 保存原始索引

    # 显示结果
    # print(f"successfully select {len(selected_variants)} new variants for next round:")
    # print(df_next_round.head)

    return df_next_round, df_pre_all_sorted

