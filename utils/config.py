import torch


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cpu = "cpu"
    
    data_path = './data/ocr/'
    checkpoint_path = './svaed/best_model_checkpoint.pth'
    dataloader_path = './saved/dataloader.pkl'
    metric_path = './saved/metric.xlsx'

    epoch = 1
    train_batch_size = 2
    val_batch_size = 1

    learning_rate = 1e-4
    momentum = 0.9
    maxlen = 512

    color = "#73a621"
    
