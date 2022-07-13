

```
data token describe

count    134000.000000
mean        682.040463
std         297.782816
min          17.000000
25%         408.000000
50%         657.000000
75%         995.000000
max        1886.000000
```





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

0.863



dev5

```
cls 토큰
parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--maxlen", type=int, default=256)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--pause", type=int, default=0)
parser.add_argument("--model_path", type=str, default='klue/roberta-large')
```

0.88



dev6

```
lstm pooler
parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--maxlen", type=int, default=256)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--pause", type=int, default=0)
parser.add_argument("--model_path", type=str, default='klue/roberta-large')
```

0.865



dev15

```
mean pooling
parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--maxlen", type=int, default=256)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--pause", type=int, default=0)
parser.add_argument("--model_path", type=str, default='klue/roberta-large')
```

checkpoint3 0.87966666 checkpoint2 0.89

dev 16

```
cls 토큰, init weight
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

dev 19 no augmentation

```
cls 토큰, gamma 2=> 3
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

dev 20 no augmentation

```
cls 토큰, gamma 2.5
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



dev 26 no augmentation

```
mean pooling, dropout1~3
parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--maxlen", type=int, default=256)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--pause", type=int, default=0)
parser.add_argument("--model_path", type=str, default='klue/roberta-large')
```

dev25 no aug

```
dev26 + alpha =0.25
```

dev26

```
dev15 + alpha=0.25
```

0.887



dev 29

```
dev15, alpha=0.25, drop=0
```



dev 33 data aug, gamma=2.2

0.9

dev 42 data integration

dev 43 dev33, gamma=2.1

dev44, gamma=2.3



dev 80 max_len = 288

dev81 max_len = 240

dev86 new data augmentation

dev88 add group parameter

dev 89 max_len 320
