import pickle
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import DataLoader

from utils.config import Config
from utils.dataset import CustomDataset

def prepare_dataset_and_dataloader(df:pd.DataFrame, tokenizer) -> Dict[str, DataLoader]:
    """
    This function prepares and returns train, validation, and test data loaders for a given dataset and
    saves them as a pickle file.
    
    :param df: A pandas DataFrame containing the data to be used for training, validation, and testing
    :param tokenizer: The tokenizer is an object that is used to convert text into numerical tokens that
    can be fed into a machine learning model. 
    :return: a dictionary containing three PyTorch DataLoader objects for the train, validation, and
    test sets respectively. 
    """
    
    np.random.seed(200)

    # Split the data into train and remaining
    train_size = 0.8
    train_df = df.sample(frac=train_size)
    remaining_df = df.drop(train_df.index)

    # Split the remaining data into val and test
    val_size = 0.5  # 10% of the original data
    val_df = remaining_df.sample(frac=val_size)
    test_df = remaining_df.drop(val_df.index)

    # Reset the indices of the dataframes
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_dataset = CustomDataset(train_df, tokenizer)
    val_dataset = CustomDataset(val_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)
    
    train_data_loader = DataLoader(train_dataset, 
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=0
    )

    val_data_loader = DataLoader(val_dataset, 
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=0
    )
    test_data_loader = DataLoader(test_dataset, 
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=0
    )
    
    dataloader = {"train":train_data_loader, "val":val_data_loader, "test":test_data_loader}
    
    #save dataloader
    with open(Config.dataloader_path, "wb") as f:
        pickle.dump(dataloader, f)
    
    return dataloader
    