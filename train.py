import re
import os
import time
import torch
# import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from typing import Dict
from utils.config import Config
from models.model import BERTClassModel
from utils.dataloader import prepare_dataset_and_dataloader

def train(train_loader:DataLoader, model:nn.Module, crtierion, optimizer:torch.optim) -> None:
   
    best_loss = float('inf')  # Initialize with a large value
    best_accuracy = 0.0
    best_model_state = None
    
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []
    
    start = time.time()
    
    for e in (range(Config.epoch)):
        
        print(f'Epoch: {e+1}')
        for mode in ["train", "val"]:
            if mode=="train":
                model.train()
            else:
                model.eval() #no update in gradients
            running_loss = 0.0
            running_acc = 0.0
            for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), colour=Config.color):
                
                ids = data['input_ids'].to(Config.device, dtype = torch.long)
                mask = data['attention_mask'].to(Config.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(Config.device, dtype = torch.long)
                targets = data['targets'].to(Config.device)

                outputs = model(ids, mask, token_type_ids)

                optimizer.zero_grad()
                outputs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, targets.squeeze())
                
                optimizer.zero_grad()
                if model=='train':
                    loss.backward()
                    optimizer.step()
                
                running_loss+=loss.item()
                
                #Accuracy
                _, max_idx = torch.max(outputs, 1)
                acc = torch.sum(max_idx==targets)
                running_acc += acc
            
            # train_loss += (loss.item() - train_loss) / (batch_idx + 1)
            if mode=="train":
                
                train_loss = running_loss  / (Config.train_batch_size*len(dataloader[mode]))
                train_acc = 100 * running_acc / (Config.train_batch_size*len(dataloader[mode]))
                
                train_losses.append(train_loss)
                train_accuracy.append(train_acc)
                
                print(f'Training Loss: {train_loss:.3f} Training Accuracy: {train_acc:.2f}%')
            else:
                
                val_loss = running_loss / (Config.val_batch_size*len(dataloader[mode]))
                val_acc = 100 * running_acc / (Config.val_batch_size*len(dataloader[mode]))
                
                print(f'Validation Loss: {val_loss:.3f} Validation Accuracy: {val_acc:.2f}%')
                
                val_losses.append(val_loss)
                val_accuracy.append(val_acc)
                
                if val_loss < best_loss or val_acc > best_accuracy:
                    best_loss = val_loss
                    best_accuracy = val_acc
                    best_model_state = model.state_dict()        
        print(f'--------------------------------------------------')
    end = time.time()
    training_time = end-start
    print(f'Training Completed in: {training_time//60} min {training_time%60:.2f} sec')
    print('Finished Training')
    #Checkpoint model
    torch.save({
            'epoch': Config.epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            }, Config.checkpoint_path)

def preprocess_text(text):
    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z0-9.,@ ]", "", text)

    # Remove new lines
    text = text.replace("\n", " ")

    # Remove empty lines
    text = "\n".join(line for line in text.split("\n") if line.strip() != "")

    # Trim leading and trailing spaces
    text = text.strip()

    return text

def load_text_in_dataframe() -> pd.DataFrame:
    """
    This function loads text data from a specified path into a pandas DataFrame and maps the class
    labels to numerical values.
    :return: a DataFrame that contains two columns: "context" and "class". 
    """
    
    data = []
    #reading through all text under ocr path with its respective class 
    for path in os.listdir(Config.data_path):
        for path_text in os.listdir(Config.data_path+path+"/"):
            with open(Config.data_path+path+"/"+path_text) as f:
                context = f.read()
                data.append((context, path))
                
    df = pd.DataFrame.from_records(
                            data, 
                            columns=["context", "class"]
                        )
    class_map = {"0": 0, "2":1, "4":2, "6":3, "9":4} #mapping
    df["class"] = df["class"].map(class_map)
    df["context"] = df["context"].apply(preprocess_text)
    return df
    

if __name__ == "__main__":
    
    df = load_text_in_dataframe()
    
    #load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataloader = prepare_dataset_and_dataloader(df, tokenizer)
    
    
    model = BERTClassModel(df["class"].nunique())
    model.to(Config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=Config.learning_rate)
    
    train(dataloader["train"], model, criterion, optimizer)
    