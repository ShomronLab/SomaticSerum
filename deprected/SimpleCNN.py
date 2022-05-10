import os, sys, re
import argparse
import time
import random
import logging

import matplotlib.pyplot as plt
from torch._C import device
from torch.utils.data import DataLoader
import torch

from src.bam_dataloader import CustomBamDataset
from src.loader import CustomBamDataset2
import src.models as Model
import src.plots as Plot

N_CHANNELS = 3

models = {
    'nucSimpleCnn': Model.nucSimpleCnn(input_size = 200, hidden_size = 64, dropout = 0.5, num_classes = 2),
    'SimpleCnn': Model.SimpleCnn(input_size = 200, hidden_size = 64, dropout = 0.5, num_classes = 2),
    'CnnLinear': Model.CnnLinear(input_size = 200, hidden_size = 64, linear_size = 8, dropout = 0.5, num_classes = 2),
    'nucSimpleCnnOrig': Model.nucSimpleCnnOrig(input_size = 200, hidden_size = 64, dropout = 0.65, num_classes = 2),
    'LSTM_Classifier': Model.LSTM_Classifier(input_size = 200, hidden_size = 64, dropout = 0.65, num_classes = 2, n_layer=10)
    # 'AttentionNetwork': Model.AttentionNetwork(input_size = 200, hidden_size = 64, dropout = 0.65, num_classes = 2, n_layer=10)
}


def genLogger(out):
    logging.basicConfig(filename = "{}/ExtractReads.{}.log".format(out,luad),
                        filemode = "a",
                        format = "%(levelname)s %(asctime)s - %(message)s", 
                        level = logging.DEBUG)
    logger = logging.getLogger()
    logger.info("Started ExtractReads.py")


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu', out='.'):
    print('\ntrain() called: model=%s, opt=%s(lr=%f), epochs=%d, dropout=%f device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, round(model.dropout, 3), device))
    name = '{}/model={}.channel={}.opt={}(lr={}).epochs={}.dropout={}.device={}.'.format(out, 
            type(model).__name__, N_CHANNELS, type(optimizer).__name__, optimizer.param_groups[0]['lr'], epochs, round(model.dropout, 3), device)
    f = open("{}txt".format(name), 'a')
    f.write('Epoch,train_loss,train_acc,val_loss,val_acc')

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:
            optimizer.zero_grad()

            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            y    = batch[2].to(device)
            yhat  = model(x1, x2)
            # print(yhat.shape, y.shape)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x1.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x1.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:
            x1   = batch[0].to(device)
            x2   = batch[1].to(device)
            y    = batch[2].to(device)
            yhat = model(x1, x2)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x1.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        # if epoch == 1 or epoch % 2 == 0:
        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
        f.write('\n%d,%5.2f,%5.2f,%5.2f,%5.2f' % (epoch, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
    f.close()

    return history, name


def split_data(data_path, train_size=0.8, val_size=0.1, test_size=0.1):
    
    bamsbai = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    bamsbai = [f for f in bamsbai if f.endswith('.bam') or f.endswith('.bai')]
    onlybams = [f for f in bamsbai if f.endswith('.bam')]
    bam_numbers = set([int(re.search(r'\d+', s).group()) for s in onlybams])
    inds = set(random.sample(list(range(len(bam_numbers))), int((1-train_size)*len(bam_numbers))))
    train_n = [n for i,n in enumerate(bam_numbers) if i not in inds]
    
    train_bams = []
    test_bams = []
    for i in train_n:
        for s in bamsbai:
            if int(re.findall(r'\d+', s)[0].lstrip('0')) == i:
                train_bams.append(s)
    for s in bamsbai:
        if s not in train_bams: test_bams.append(s)

    train_bams = [os.path.join(data_path, s) for s in train_bams]
    test_bams = [os.path.join(data_path, s) for s in test_bams]
    return train_bams, test_bams


def main(args):
    # general setting up
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    num_workers = 1
    torch.manual_seed(42)
    train_validation_split = 0.8
    length_of_reads = int(151 * 2)

    # data loading and filtering
    training_bam_dir = args.training_bam_dir

    # random bam split
    # full_train_dataset = CustomBamDataset2(training_bam_dir, whichSet = 'train')
    # train_size = int(train_validation_split * len(full_train_dataset))
    # test_size = len(full_train_dataset) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

    full_train_dataset, full_test_dataset = split_data(training_bam_dir)
    train_dataset = CustomBamDataset2(full_train_dataset, whichSet = 'train')
    valid_dataset = CustomBamDataset2(full_test_dataset, whichSet = 'test')
    train_size = len(full_train_dataset)
    test_size = len(full_test_dataset)

    print('train_size: %d, test_size: %d' % (train_size, test_size))

    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=num_workers)

    # deep learning setting up
    # nucleotide_model = Model.crnn_Classifier(N_CHANNELS, hidden_size)
    nucleotide_model = models[args.model]
    loss_fn = torch.nn.CrossEntropyLoss()
    params = list(nucleotide_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, eps=1e-08, weight_decay=0.01)

    # training
    if torch.cuda.is_available():
        nucleotide_model = nucleotide_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    history, name = train(model = nucleotide_model,
                        optimizer = optimizer,
                        loss_fn = loss_fn,
                        train_dl = train_dataloader,
                        val_dl = valid_dataloader,
                        epochs = max_epoch,
                        device = device,
                        out = args.out)

    # plotting
    Plot.plot(name)
    
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)


""" Helper function to test data loading"""
def test_bam_loading(training_bam_dir):
    example_dataset = CustomBamDataset(training_bam_dir, args.out, N_CHANNELS, bam_lines_maximum = 100)
    train_size = int(0.8 * len(example_dataset))
    test_size = len(example_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(example_dataset, [train_size, test_size])
    print(train_dataset[3])
    # print(test_dataset[3])


if __name__ == '__main__':
    # test_bam_loading("bam_train_new/")
    parser = argparse.ArgumentParser('Train the model.')
    parser.add_argument('training_bam_dir', type=str,
                        help='Train data bams directory')
    # parser.add_argument('testing_bam_dir', type=str,
    #                     help='Test data bams directory')
    parser.add_argument('--model', type=str, help='model', 
                        required=True, default='SimpleCnn')
    parser.add_argument('--hidden-size', required=True, type=int,
                        help='The number of hidden units')
    parser.add_argument('--batch-size', required=True, type=int,
                        help='The size of each batch')
    parser.add_argument('--learning-rate', required=True, type=float,
                        help='The learning rate value')
    parser.add_argument('--max-epoch', required=True, type=int,
                        help='The maximum epoch')
    # parser.add_argument('--num-workers', required=True, type=int,
    #                     help='The number of workers for parallel data loading')
    parser.add_argument('--out', required=False, type=str,
                        help='Output directory', default='output')
    args = parser.parse_args()

    args.out = os.path.join(args.training_bam_dir, args.out)
    os.makedirs(args.out, exist_ok=True)

    main(args)
    # test_bam_loading(args.training_bam_dir)

