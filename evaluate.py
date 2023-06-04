import torch
import pickle
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from utils.config import Config
from models.model import BERTClassModel

def evaluate(model, dataloader):
    
    labels = []
    predicted = []
    
    
    print("Evaluating on Test set....")
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), colour=Config.color):
            
            ids = data['input_ids'].to(Config.device, dtype = torch.long)
            mask = data['attention_mask'].to(Config.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(Config.device, dtype = torch.long)
            targets = data['targets'].to(Config.device)

            outputs = model(ids, mask, token_type_ids)
            
            outputs = torch.softmax(outputs, dim=1)
            
            #Accuracy
            _, max_idx = torch.max(outputs, 1)
            
            predicted.extend(max_idx.tolist())
            labels.extend(targets.tolist())
    
    return \
        precision_score(targets, predicted, average="macro"), \
        recall_score(targets, predicted, average="macro"), \
        accuracy_score(targets, predicted), \
        f1_score(targets, predicted, average="macro"), \
        confusion_matrix(targets, predicted)
        
if __name__ == "__main__":
    
    #load trained model
    model = BERTClassModel(5)
    checkpoint = torch.load(Config.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #load test dataloader
    with open(Config.dataloader_path, "rb") as f:
        dataloader = pickle.load(f)
    
    ps, rs, acc, f1, conf_matrix = evaluate(model.to(Config.device), dataloader["test"])
 

    # Print the metrics in a fancy format
    print(f"Precision: {ps:.4f}")
    print(f"Recall: {rs:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(tabulate(conf_matrix, tablefmt="fancy_grid"))

    
    # Save the DataFrame to an Excel file
    data = {'Precision': [ps], 'Recall': [rs], 'Accuracy': [acc], 'F1 Score': [f1], "Confusion Matrix": [conf_matrix]}
    df = pd.DataFrame(data)
    df.to_excel(Config.metric_path, index=False)