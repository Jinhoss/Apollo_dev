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
import random


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        test_dataset = ApolloDataset(file_path, args)
        test_dataLoader = DataLoader(test_dataset)
        results, _ = predict(model, args, test_dataLoader)
        return results

    nsml.bind(save, load, infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
    parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--model_path", type=str, default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path)
    args.valid_path = os.path.join(DATASET_PATH, args.valid_path)

    print(args)

    # model load
    model = ApolloModel(args)
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
        print('dataset complete', len(train_dataset))
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        valid_dataLoader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True)
        print('dataloader complete')
        train(model, args, train_dataLoader, valid_dataLoader)
