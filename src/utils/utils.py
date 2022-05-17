import logging
import os
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import modules.models as Model
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


class Parms:
    def __init__(self, args):
        self.BAM_DIR        = args.training_bam_dir
        self.MODEL          = args.model
        self.HIDDEN         = args.hidden_size
        self.DROPOUT        = args.dropout
        self.BATCH_SIZE     = args.batch_size
        self.LEARNING_RATE  = args.learning_rate
        self.MAX_EPOCH      = args.max_epoch
        self.NUM_WORKERS    = args.num_workers
        self.TEST           = args.test
        self.SAMPLE_SPLIT   = args.sample_split
        self.LSTM_LAYERS    = args.lstm_layers
        self.INPUT_SIZE     = args.sequence_length

        self.OUT            = self.return_output_dir(args)

        self.NUM_CLASSES    = 2
        self.N_CHANNELS     = 3
        self.LINEAR_SIZE    = 8
        self.TRAIN_VALIDATION_SPLIT = 0.8

        self.genLogger()

    def return_model(self):
        Models = {
            'nucSimpleCnn':         Model.nucSimpleCnn(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN , dropout = self.DROPOUT, num_classes = self.NUM_CLASSES),
            'SimpleCnn':            Model.SimpleCnn(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, dropout = self.DROPOUT, num_classes = self.NUM_CLASSES),
            'CnnLinear':            Model.CnnLinear(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, linear_size = self.LINEAR_SIZE, dropout = self.DROPOUT, num_classes = self.NUM_CLASSES),
            'nucSimpleCnnOrig':     Model.nucSimpleCnnOrig(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, dropout = self.DROPOUT, num_classes = self.NUM_CLASSES),
            'LSTM_Classifier':      Model.LSTM_Classifier(num_classes = self.NUM_CLASSES, input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, num_layers = self.LSTM_LAYERS, seq_length = self.INPUT_SIZE),
            'ConvLSTM_Classifier':  Model.ConvLSTM_Classifier(num_classes = self.NUM_CLASSES, input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, num_layers = self.LSTM_LAYERS, seq_length = self.INPUT_SIZE),
            'AttentionNetwork':     Model.AttentionNetwork(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN, k_max = 20, dropout = self.DROPOUT)
                }
        return Models[self.MODEL]
    
    def return_output_dir(self, args):
        out = os.path.join(self.BAM_DIR, "{}/{}_epoch{}dropout{}hidden{}batchsize{}_{}".format(args.out, self.MODEL, 
                self.MAX_EPOCH, self.DROPOUT, self.HIDDEN, time.strftime("%d-%m*%H:%M"), self.BATCH_SIZE))
        os.makedirs(out, exist_ok=True)
        return out

    def genLogger(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename= os.path.join(self.OUT, 'train.log'),
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        logging.info('Logger initialized')


def split_data(data_path, train_size=0.8, val_size=0.1, test_size=0.1):
    
    try:
        assert train_size + val_size + test_size == 1
    except AssertionError:
        logging.error("train_size + val_size + test_size should be equal to 1")
        sys.exit(1)

    bamsbai = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    bamsbai = [f for f in bamsbai if f.endswith('.bam') or f.endswith('.bai')]
    onlybams = [f for f in bamsbai if f.endswith('.bam')]
    bam_numbers = list(set([int(re.search(r'\d+', s).group()) for s in onlybams]))
    random.shuffle(bam_numbers)

    n_train = int(train_size * len(bam_numbers))
    n_val = int(val_size * len(bam_numbers))
    n_test = int(test_size * len(bam_numbers))
    
    random.shuffle(bam_numbers)
    train_samples = bam_numbers[:n_train]
    val_samples = bam_numbers[n_train:n_train+n_val]
    test_samples = bam_numbers[n_train+n_val:]
    logging.info("Samples: {}".format(bam_numbers))
    
    # train_samples = [30, 6, 11, 29, 18, 13, 3, 5, 24, 26, 33, 8, 21, 1, 2, 9, 12, 20, 34, 17, 7, 4]
    # val_samples = [16, 25, 19, 10]
    # test_samples = [32, 31]

    logging.info("train_samples: {}, count {}".format(train_samples, len(train_samples)))
    logging.info("val_samples: {}, count {}".format(val_samples, len(val_samples)))
    logging.info("test_samples: {}, count {}".format(test_samples, len(test_samples)))
    
    train_bams = []
    val_bams = []
    test_bams = []
    for bambai in bamsbai:
        i = int(re.findall(r'\d+', bambai)[0].lstrip('0')) 
        if i in train_samples:
            train_bams.append(bambai)
        elif i in val_samples:
            val_bams.append(bambai)
        elif i in test_samples:
            test_bams.append(bambai)
        else:
            logging.error("Error in splitting data {}".format(bambai))
            sys.exit(1)
    # print("train_bams: ", len(train_bams))
    # print("val_bams: ", len(val_bams))
    # print("test_bams: ", len(test_bams))
    return train_bams, val_bams, test_bams


def get_classification_success_stats(epoch, cm, clas):
    """ Confusion matrix, Accuracy, sensitivity and specificity"""   
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    
    tpr = round(tp/(tp+fn), 3)            # Sensitivity, hit rate, recall, or true positive rate
    tnr = round(tn/(tn+fp), 3)            # Specificity or true negative rate
    ppv = round(tp/(tp+fp), 3)            # Precision or positive predictive value
    npv = round(tn/(tn+fn), 3)            # Negative predictive value
    fpr = round(fp/(fp+tn), 3)            # Fall out or false positive rate
    fnr = round(fn/(tp+fn), 3)            # False negative rate
    fdr = round(fp/(tp+fp), 3)            # False discovery rate
    acc = round((tp+tn)/(tp+fp+fn+tn), 3) # Overall accuracy for each class
    f1 = round(2*((tpr*ppv)/(tpr+ppv)), 3)
 
    return epoch, tpr, tnr, ppv, npv, fpr, fnr, fdr, acc, f1, clas


def plot(input_file, metric, metric_test, out):
    name = ' '.join(input_file.split('/')[-1].split('.'))
    df = pd.read_csv("{}txt".format(input_file))
    df.set_index('Epoch', inplace=True)

    with plt.style.context('bmh'):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx() 

        df['train_acc'].plot(color='b', linestyle = 'solid', ax=ax2, label='train_acc')
        df['val_acc'].plot(color='b', linestyle = '-.', ax=ax2, label='val_acc')
        df['train_loss'].plot(color='g', linestyle = 'solid', ax=ax1, label='train_loss')
        df['val_loss'].plot(color='g', linestyle = '-.', ax=ax1, label='val_loss')

        metric = metric.iloc[-2:]
        metric.set_index('Epoch', inplace=True)
        concat_metric = pd.concat([metric, metric_test], ignore_index=True)
        concat_metric.drop({'Epoch'}, axis=1, inplace=True)
        concat_metric.set_index('Class', inplace=True)
        plt.table(cellText=concat_metric.values, colWidths=[0.095]*len(concat_metric.columns),
                    rowLabels=concat_metric.index,
                    colLabels=concat_metric.columns,
                    cellLoc = 'center', rowLoc = 'center',
                    loc='top')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='g')
        ax1.set(ylim=(0.4, 0.9))
        ax1.legend(loc='upper left')
        ax2.set_ylabel('Accuracy', color='b')
        ax2.set(ylim=(0.4, 0.9))
        ax2.legend(loc='upper right')

        fig.suptitle(name)

        plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)
        fig.set_size_inches(17.5, 9.5)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("{}/plot.png".format(out)); # shows the plot. 

if __name__ == "__main__":
    pass
    # input_file = sys.argv[1]
    # metric = pd.read_csv(sys.argv[2])
    
    # plot(input_file, metric)

    # # split data
    # data_path = '/data/hadasvol/projects/cancer_plasma/seqmerge/DLbams_rand'
    # split_data(data_path)

    # test Parms class
    # import argparse
    # import time

    # parser = argparse.ArgumentParser('Train the model.')
    # parser.add_argument('training_bam_dir', type=str,
    #                     help='Train data bams directory')
    # parser.add_argument('--sample_split', type=str,
    #                     help='How to split the training data: True - by samples, False - by random on the entire dataset',
    #                     default = 'True')
    # parser.add_argument('--model', type=str, help='model', 
    #                     required=True, default='SimpleCnn')
    # parser.add_argument('--hidden-size', required=True, type=int,
    #                     help='The number of hidden units')
    # parser.add_argument('--batch-size', required=True, type=int,
    #                     help='The size of each batch')
    # parser.add_argument('--learning-rate', required=True, type=float,
    #                     help='The learning rate value')
    # parser.add_argument('--max-epoch', required=True, type=int,
    #                     help='The maximum epoch')
    # parser.add_argument('--out', required=False, type=str,
    #                     help='Output directory', default='output')
    # parser.add_argument('--test', required=False, type=str,
    #                     help='Test directory')
    # args = parser.parse_args()

    # args.out = os.path.join(args.training_bam_dir, "{}_{}_{}".format(args.model, time.strftime("%d-%m*%H:%M"), args.out))

    # parms = Parms(args)
