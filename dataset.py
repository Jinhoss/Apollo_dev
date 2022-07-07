from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import re
from transformers import AutoTokenizer
import torch

class ApolloDataset(Dataset):
    def __init__(self, file_path, args):
        super(ApolloDataset, self).__init__()
        self.df = pd.read_csv(file_path)
        self.tokenizer=  AutoTokenizer.from_pretrained(args.model_path, do_lower_case=False)
        self.df['contents'] = self.df['contents'].apply(self.cleansing)
        self.args = args
        if args.mode=='train':
            self.df = self.dataAugmentation(self.df, self.tokenizer)

    def dataAugmentation(self, sen_df, tokenizer):
        positive = sen_df[sen_df['label']==0]
        negative = sen_df[sen_df['label']==1]
        negative.reset_index(drop=True)
        positive = positive.sample(frac=1).reset_index(drop=True)
        l = len(sen_df)//2 - len(negative)
        new_sen = []
        for i in range(l//2):
            idx = i*2
            sen1 = positive['contents'].iloc[idx]
            sen2 = positive['contents'].iloc[idx+1]
            sen1 = tokenizer.tokenize(sen1)
            sen2 = tokenizer.tokenize(sen2)
            min_len = min(len(sen1), len(sen2))
            new1, new2 = [], []
            for j in range(min_len):
                if j%2:
                    new1.append(sen1[j])
                    new2.append(sen2[j])
                else:
                    new2.append(sen1[j])
                    new1.append(sen2[j])
            new1 = tokenizer.convert_tokens_to_string(new1)
            new2 = tokenizer.convert_tokens_to_string(new2)
            new_sen.append(new1)
            new_sen.append(new2)
        new_label = [1] * len(new_sen)
        insert_df = pd.DataFrame({'contents':new_sen, 'label':new_label})
        positive2 = positive[len(new_sen):]
        positive2 = positive2.reset_index(drop=True)
        add_df = positive2.append(insert_df, ignore_index=True)
        add_df = add_df.append(negative, ignore_index=True)
        return add_df
    
    def cleansing(self, sen):
        sen = re.sub('[^a-z0-9ㄱ-힣]', ' ', sen)
        sen = re.sub(' +', ' ', sen)
        return sen

    def __getitem__(self, idx):
        if self.args.mode=='train':
            sen, label = self.df['contents'][idx], self.df['label'][idx]
            label = torch.LongTensor([label])
        else:
            sen = self.df['contents'][idx]
            label = torch.LongTensor([-1])
        output = self.tokenizer.encode_plus(sen, max_length=self.args.maxlen, truncation=True, padding='max_length')
        input_ids, attention_mask = output['input_ids'], output['attention_mask']
        input_ids, attention_mask = torch.LongTensor(input_ids), torch.LongTensor(attention_mask)

        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.df)



