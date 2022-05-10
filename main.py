import os, sys, re
import argparse
import time
import random
import logging

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch._C import device
from torch.utils.data import DataLoader
import torch

from train import train
from test import test
from src.loader import CustomBamDataset2
import src.util as util
from src.util import Parms


def main(args):
    # --- INITIALIZE INPUT PARMATERS -------------------------------------
    torch.manual_seed(42)
    parms = Parms(args)
    for key, val in vars(parms).items():
        logging.info('%s: %s' % (key, val))

    # --- Data loading and filtering -------------------------------------
    if parms.SAMPLE_SPLIT == 'True':
        full_train_dataset, full_val_dataset, full_test_dataset = util.split_data(parms.BAM_DIR)
        train_dataset   = CustomBamDataset2(full_train_dataset, read_length = parms.INPUT_SIZE,  out = parms.OUT, whichSet = 'trainSampleSplit', input_files_path = parms.BAM_DIR)
        valid_dataset   = CustomBamDataset2(full_val_dataset, read_length = parms.INPUT_SIZE, out = parms.OUT, whichSet = 'valSampleSplit', input_files_path = parms.BAM_DIR)
        test_dataset    = CustomBamDataset2(full_test_dataset, read_length = parms.INPUT_SIZE, out = parms.OUT, whichSet = 'testSampleSplit', input_files_path = parms.BAM_DIR)
        train_size      = len(full_train_dataset)
        val_size        = len(full_val_dataset)
        test_size       = len(full_test_dataset)
    else:
        # random bam split
        full_train_dataset  = CustomBamDataset2(parms.BAM_DIR, out = parms.OUT, whichSet = 'train')
        train_size          = int(parms.TRAIN_VALIDATION_SPLIT * len(full_train_dataset))
        val_size            = len(full_train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    logging.info('train_size: %d, val_size: %d, test_size: %d' % (train_size, val_size, test_size))

    if parms.TEST == None:
        train_dataloader = DataLoader(train_dataset, batch_size = parms.BATCH_SIZE, shuffle = True, num_workers = parms.NUM_WORKERS)
        valid_dataloader = DataLoader(valid_dataset, batch_size = parms.BATCH_SIZE, shuffle = True, num_workers = parms.NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size = parms.BATCH_SIZE, shuffle = True, num_workers = parms.NUM_WORKERS)

    # --- Model setup ----------------------------------------------------
    logging.info('Setting up the model...')
    nucleotide_model    = parms.return_model()
    loss_fn             = torch.nn.CrossEntropyLoss()
    model_params        = list(nucleotide_model.parameters())
    optimizer           = torch.optim.AdamW(model_params, lr=parms.LEARNING_RATE, eps=1e-08, weight_decay=0.01)

    # --- Training -------------------------------------------------------
    if torch.cuda.is_available():
        nucleotide_model = nucleotide_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    if parms.TEST:
        # Skip training
        logging.info('Skipping train')
        metric = pd.read_csv('{}/metric.csv'.format(parms.TEST))
        name = '{}/{}'.format(parms.TEST, [f for f in os.listdir(parms.TEST) if f.endswith('.txt')][0].split('txt')[0])
        model_path = "{}/{}.pth".format(parms.TEST, type(nucleotide_model).__name__)
        metric_test = test(parms, model_path, test_dataloader, device, loss_fn)
    else:
        logging.info('Training...')
        history, name, metric, model_path = train(model = nucleotide_model,
                                    optimizer = optimizer,
                                    loss_fn = loss_fn,
                                    train_dl = train_dataloader,
                                    val_dl = valid_dataloader,
                                    epochs = parms.MAX_EPOCH,
                                    device = device,
                                    out = parms.OUT)
        # Test
        metric_test = test(parms, model_path, test_dataloader, device, loss_fn)
    
    # --- Plotting -------------------------------------------------------
    logging.info('Plotting...')
    util.plot(name, metric, metric_test, parms.OUT)
    
    # acc = history['acc']
    # val_acc = history['val_acc']
    # loss = history['loss']
    # val_loss = history['val_loss']
    # epochs = range(1, len(acc) + 1)

    logging.info("Done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or test SomaticSerum model.')
    parser.add_argument('training_bam_dir', type=str,
                        help='Train data bams directory')
    parser.add_argument('--sample_split', required=False, type=str,
                        help='How to split the training data: True - by samples, False - by random on the entire dataset',
                        default = 'True')
    parser.add_argument('--model', required=False, type=str, 
                        help='model', default='SimpleCnn')
    parser.add_argument('--hidden_size', required=False, type=int,
                        help='The number of hidden units', default=64)
    parser.add_argument('--sequence_length', required=False, type=int,
                        help='The length of the sequence', default=200)
    parser.add_argument('--batch_size', required=False, type=int,
                        help='The size of each batch', default=512)
    parser.add_argument('--learning_rate', required=False, type=float,
                        help='The learning rate value', default=0.00001)
    parser.add_argument('--max_epoch', required=False, type=int,
                        help='The maximum epoch', default=100)
    parser.add_argument('--lstm_layers', required=False, type=int,
                        help='Num of LSTM layers', default=10)
    parser.add_argument('--dropout', required=False, type=float,
                        help='Dropout', default=0.5)
    parser.add_argument('--num_workers', required=False, type=int,
                        help='Number of workers', default=1)
    parser.add_argument('--out', required=False, type=str,
                        help='Output directory', default='output')
    parser.add_argument('--test', required=False, type=str,
                        help='Test directory')
    args = parser.parse_args()

    main(args)

    # @dataclass
    # class Args:
    #     training_bam_dir: str = "/data/hadasvol/projects/cancer_plasma/seqmerge/DLbams_rand"
    #     sample_split: bool = True
    #     model: str = "CnnLinear"
    #     hidden_size: int = 64
    #     batch_size: int = 512
    #     learning_rate: float = 0.00001
    #     max_epoch: int = 100
    #     out: str = os.path.join("/data/hadasvol/projects/cancer_plasma/seqmerge/DLbams_rand", "{}_{}_{}".format("CnnLinear", time.strftime("%d-%m*%H:%M"), 'output'))
    
    # args = Args()
    
    # for _ in range(5):
    #     main(args)
    #     os.remove(os.path.join(args.training_bam_dir, "data_trainSampleSplit.pkl"))
    #     os.remove(os.path.join(args.training_bam_dir, "data_testSampleSplit.pkl"))

