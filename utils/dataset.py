import torch

from utils.config import Config

class CustomDataset(torch.utils.data.Dataset):
    '''
        This is a custom dataset class for tokenizing and encoding text data with a bert tokenizer and
        returns encoded text and its respective target.
    
    '''
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.df = df
        self.title = self.df['context']
        self.targets = self.df["class"].values.astype('int')

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=Config.maxlen,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.LongTensor([self.targets[index]])
        }