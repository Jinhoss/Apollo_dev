dev1

```
cls 토큰
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
```

0.8603



dev3

```
lstm pooler
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
```

