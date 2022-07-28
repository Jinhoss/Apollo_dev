import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import nsml
from loss import FocalLoss


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def predict(model, args, data_loader):
    print('start predict')
    model.eval()

    eval_accuracy = []
    logits = []
    
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(args.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)
        logit = outputs
        logit = logit.detach().cpu().numpy()
        label = b_labels.cpu().numpy()

        logits.append(logit)

        accuracy = flat_accuracy(logit, label)
        eval_accuracy.append(accuracy)

    logits = np.vstack(logits)
    predict_labels = np.argmax(logits, axis=1)
    return predict_labels, np.mean(eval_accuracy)


def train(model, args, train_loader, valid_loader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      eps=args.eps
                      )
    total_steps = len(train_loader) * args.epochs
    criterion = FocalLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    print('start training')
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        gradient_accumulation_steps = args.steps
        for step, batch in enumerate(train_loader):
            model.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)
            y = b_labels.view(-1)
            loss = criterion(outputs, y)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            train_loss.append(loss.item())
            loss.backward()
            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        avg_train_loss = np.mean(train_loss)
        _, avg_val_accuracy = predict(model, args, valid_loader)
        print("Epoch {0},  Average training loss: {1:.4f} , Validation accuracy : {2:.4f}"\
              .format(epoch, avg_train_loss, avg_val_accuracy))

        nsml.save(epoch)
    return model
