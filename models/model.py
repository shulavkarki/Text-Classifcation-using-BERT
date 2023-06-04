import torch
from transformers import BertModel 


class BERTClassModel(torch.nn.Module):
    ''' BERT model(encoder) with a linear layer for mulit-class classification.
    '''
    def __init__(self, n_class):
        super(BERTClassModel, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, n_class)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output