import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import nsml
from transformers import AutoModel


class ApolloModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(args.model_path)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(self.bert_model.config.hidden_size, (self.bert_model.config.hidden_size) // 2, num_layers=2, 
                              dropout=0.1, batch_first=True,
                              bidirectional=True)
        self.pooler = MeanPooling()
        # self._init_params()
        
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)


    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.pooler(output.last_hidden_state, attention_mask)
        # output = output[0][:, 0, :]
        # output = self.bilstm(output.last_hidden_state)
        # output = torch.sum(output[0], dim=1)/output[0].size()[1]
        # logit1 = self.fc(self.dropout1(output))
        # logit2 = self.fc(self.dropout2(output))
        # logit3 = self.fc(self.dropout3(output))
        return self.fc(output)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
