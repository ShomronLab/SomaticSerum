import time, logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch._C import device
import torch

import src.util as util

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu', out='.'):
    logging.info('train() called: model=%s, opt=%s(lr=%f), epochs=%d, dropout=%f device=%s' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, round(model.dropout, 3), device))
    name = '{}/model={}.opt={}(lr={}).epochs={}.dropout={}.device={}.'.format(out, 
            type(model).__name__, type(optimizer).__name__, optimizer.param_groups[0]['lr'], epochs, round(model.dropout, 3), device)
    f = open("{}txt".format(name), 'a')
    f.write('Epoch,train_loss,train_acc,val_loss,val_acc')

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    metric = pd.DataFrame(columns=['Epoch', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'Accuracy', 'F1', 'Class'])

    best_val_acc = 0
    model_path = '{}/{}.pth'.format(out, type(model).__name__)
    start_time_sec = time.time()
    for epoch in range(1, epochs+1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        CM_train=0

        for batch in train_dl:
            optimizer.zero_grad()

            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            y    = batch[2].to(device)
            yhat  = model(x1, x2)
            # logging.info(yhat.shape, y.shape)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x1.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x1.shape[0]

            labels = y
            outputs = yhat
            preds = torch.argmax(outputs.data, 1)
            CM_train+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)
        metric.loc[len(metric)] = list(util.get_classification_success_stats(epoch, CM_train, 'train'))

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0
        CM_val=0

        for batch in val_dl:
            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            y    = batch[2].to(device)
            yhat = model(x1, x2)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x1.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

            labels = y
            outputs = yhat
            preds = torch.argmax(outputs.data, 1)
            CM_val+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)
        cur_val_metric = list(util.get_classification_success_stats(epoch, CM_val, 'val'))
        metric.loc[len(metric)] = cur_val_metric

        # if epoch == 1 or epoch % 2 == 0:
        logging.info('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
        if epoch != epochs:
            f.write('\n%d,%5.2f,%5.2f,%5.2f,%5.2f' % (epoch, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # SAVE MODEL
        if epoch > epochs/2 and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_metric = cur_val_metric
            try:
                torch.save(model.state_dict(), model_path)
                logging.info('Saved model with best validation accuracy: %5.4f' % best_val_acc)
            except:
                logging.info('Failed to save model with best validation accuracy: %5.4f' % best_val_acc)
    
    # END OF TRAINING LOOP
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    logging.info('Time total:     %5.2f sec' % (total_time_sec))
    logging.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    f.write('\n%d,%5.2f,%5.2f,%5.2f,%5.2f' % (epoch, train_loss, train_acc, best_val_loss, best_val_acc))
    history['loss'].append(train_loss)
    history['val_loss'].append(best_val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(best_val_acc)
    f.close()

    metric.loc[len(metric)-1] = best_val_metric
    metric.to_csv('{}/metric.csv'.format(out), index=False)

    return history, name, metric ,model_path
