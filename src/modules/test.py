import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utils.utils as util
from sklearn.metrics import confusion_matrix
from torch._C import device


# Function to test the model 
def test(parms, model_path, test_dl, device, loss_fn):
    # --- EVALUATE ON TEST SET -------------------------------------
    model = parms.return_model()
    try:
        model.load_state_dict(torch.load(model_path))
        logging.info('Loaded model from %s' % model_path)
        model = model.cuda() if device == 'cuda' else model.cpu()
    except:
        logging.info('Failed to load model from %s' % model_path)
        sys.exit(1)

    test_loss         = 0.0
    num_test_correct  = 0
    num_test_examples = 0
    CM_test           = 0
    metric = pd.DataFrame(columns=['Epoch', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'Accuracy', 'F1', 'Class'])

    for batch in test_dl:
        x1   = batch[0].to(device)
        x2   = batch[1].to(device)
        y    = batch[2].to(device)
        yhat = model(x1, x2)
        loss = loss_fn(yhat, y)

        test_loss         += loss.data.item() * x1.size(0)
        num_test_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
        num_test_examples += y.shape[0]

        labels  = y
        outputs = yhat
        preds   = torch.argmax(outputs.data, 1)
        CM_test += confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

    test_acc  = num_test_correct / num_test_examples
    test_loss = test_loss / len(test_dl.dataset)
    metric.loc[len(metric)] = list(util.get_classification_success_stats(None, CM_test, 'test'))
    metric.to_csv('{}/metric.test.csv'.format(parms.OUT), index=False)

    logging.info('Test loss: %5.4f, test acc: %5.4f' % (test_loss, test_acc))

    metric_list = metric.iloc[-1].to_list()
    metric_list += [parms.MAX_EPOCH, parms.BATCH_SIZE, parms.LEARNING_RATE, parms.DROPOUT, parms.HIDDEN]
    metric_str = ','.join(map(str, metric_list)) + '\n'
    with open("{}/test.out".format(parms.BAM_DIR), "a") as test_file:
        test_file.write(metric_str)
    return metric

