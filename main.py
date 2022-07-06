import torch
import argparse
from train import train, predict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import torch
import numpy as np
import time
import datetime
import nsml
from nsml import DATASET_PATH, DATASET_NAME
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from dataset import ApolloDataset
from model import ApolloModel


# def generate_data_loader(file_path, tokenizer, args):
#     def get_input_ids(data):
#         document_bert = ["[CLS] " + str(s) + " [SEP]" for s in data]
#         tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
#         input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
#         input_ids = pad_sequences(input_ids, maxlen=args.maxlen, dtype='long', truncating='post', padding='post')
#         return input_ids

#     def get_attention_masks(input_ids):
#         attention_masks = []
#         for seq in input_ids:
#             seq_mask = [float(i > 0) for i in seq]
#             attention_masks.append(seq_mask)
#         return attention_masks

#     def get_data_loader(inputs, masks, labels, batch_size=args.batch):
#         data = TensorDataset(torch.tensor(inputs), torch.tensor(masks), torch.tensor(labels))
#         sampler = RandomSampler(data) if args.mode == 'train' else SequentialSampler(data)
#         data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
#         return data_loader

#     data_df = pd.read_csv(file_path)
#     input_ids = get_input_ids(data_df['contents'].values)
#     attention_masks = get_attention_masks(input_ids)
#     data_loader = get_data_loader(input_ids, attention_masks, data_df['label'].values)

#     return data_loader


def bind_nsml(model, args=None):
    def save(dir_name, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))

    def load(dir_name, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'), map_location=args.device)
        model.load_state_dict(state, strict=False)
        print('model is loaded')

    def infer(file_path, **kwargs):
        print('start inference')
        # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        # test_dataloader = generate_data_loader(file_path, tokenizer, args)
        test_dataset = ApolloDataset(file_path, args)
        test_sampler = SequentialSampler(test_dataset)
        test_dataLoader = DataLoader(test_dataset, sampler=test_sampler)
        results, _ = predict(model, args, test_dataLoader)
        return results

    nsml.bind(save, load, infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
    parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--model_path", type=str, default='klue/roberta-large')
    args = parser.parse_args()

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path)
    args.valid_path = os.path.join(DATASET_PATH, args.valid_path)

    print(args)

    # model load
    model = ApolloModel(args)
    # model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    model.to(args.device)

    bind_nsml(model, args=args)

    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":
        data_df = pd.read_csv(args.train_path)
        train_dataset = ApolloDataset(args.train_path, args)
        valid_dataset = ApolloDataset(args.valid_path, args)
        print('dataset complete')
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)
        train_dataLoader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch)
        valid_dataLoader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.batch)
        print('dataloader complete')
        # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        # train_dataloader = generate_data_loader(args.train_path, tokenizer, args)
        # validation_dataloader = generate_data_loader(args.valid_path, tokenizer, args)
        train(model, args, train_dataLoader, valid_dataLoader)
