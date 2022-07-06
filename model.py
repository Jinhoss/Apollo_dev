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
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(output[0][:, 0, :]))