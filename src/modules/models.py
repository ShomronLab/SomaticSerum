import sys

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional


class metaSimpleDense(nn.Module):
    def __init__(self, input_size, hidden_size=10, n_layers=1, dropout=0.005, num_classes=2, **kwargs):
        super(metaSimpleDense, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.num_classes = num_classes

        self.drop = nn.Dropout(p=self.dropout, inplace=False)
        self.init_fc = nn.Linear(in_features=input_size,
                                 out_features=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=hidden_size)
        self.last_fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes
        )

    def forward(self, x_meta):
        # input_nuc = torch.unsqueeze(x,1)
        # batchSize = input_nuc.size(0)

        output = self.init_fc(x_meta)
        output = functional.sigmoid(output)

        for _ in range(self.n_layers):
            output = self.fc(output)
            output = functional.sigmoid(output)
            output = self.drop(output)

        output = self.last_fc(output)
        # output = functional.relu(output)

        # output = self.drop(output)

        # output = torch.flatten(output, 1)
        return functional.softmax(output, dim=-1)  # [:, 0]


class nucSimpleCnnOrig(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.005, num_classes=2, **kwargs):
        super(nucSimpleCnnOrig, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.conv1d1 = nn.Conv1d(
            4, hidden_size, 11, stride=1, padding=5, bias=True)

        self.conv1d2 = nn.Conv1d(
            hidden_size, hidden_size, 7, stride=1, padding=3, groups=1, bias=True)

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv1d3 = nn.Conv1d(
            hidden_size, hidden_size * 8, 7, stride=1, padding=3, groups=1, bias=True)

        self.drop = nn.Dropout(p=self.dropout, inplace=False)

        self.layerNorm = nn.BatchNorm1d(int(hidden_size * 8))

        self.fc = nn.Linear(in_features=int(hidden_size * 4 * self.input_size),
                            out_features=int((hidden_size * 2) + hidden_size))

        self.fc2 = nn.Linear(in_features=int((hidden_size * 2) + hidden_size),
                             out_features=int(num_classes))

    def forward(self, input_nuc, length=None, hx=None):
        # input_nuc = torch.unsqueeze(x,1)
        # batchSize = input_nuc.size(0)

        output = self.conv1d1(input_nuc)
        output = functional.relu(output)

        output = self.conv1d2(output)
        output = functional.relu(output)
        output = self.pool2(output)

        output = self.conv1d3(output)
        output = functional.relu(output)

        output = self.drop(output)
        output = self.layerNorm(output)

        output = torch.flatten(output, 1)
        output = functional.relu(self.fc(output))

        return functional.softmax(self.fc2(output), dim=-1)


class nucSimpleCnn(nn.Module):
    def __init__(self, input_size=200, hidden_size=64, dropout=0.5, num_classes=2, **kwargs):
        super(nucSimpleCnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.conv1d1 = nn.Conv1d(
            1, hidden_size, 11, stride=5, padding=1, bias=True)

        self.conv1d2 = nn.Conv1d(
            hidden_size, hidden_size, 1, stride=1, padding=1, groups=1, bias=True)

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv1d3 = nn.Conv1d(
            hidden_size, hidden_size * 8, 7, stride=1, padding=3, groups=1, bias=True)

        self.drop = nn.Dropout(p=self.dropout, inplace=False)

        self.layerNorm = nn.BatchNorm1d(int(hidden_size * 8))

        self.fc = nn.Linear(in_features=int(hidden_size * 160),
                            out_features=int((hidden_size * 3) + hidden_size))

        self.fc2 = nn.Linear(in_features=int((hidden_size * 3) + hidden_size),
                             out_features=int(num_classes))

    def forward(self, x1, x2=None, length=None, hx=None):
        x = torch.split(x1, 1, dim=1)[1]

        output = functional.relu(self.conv1d1(x))
        output = functional.relu(self.conv1d2(output))
        output = self.pool2(output)
        output = functional.relu(self.conv1d3(output))
        output = self.drop(output)
        output = self.layerNorm(output)
        output = torch.flatten(output, 1)
        output = functional.relu(self.fc(output))
        return functional.softmax(self.fc2(output), dim=-1)


class SimpleCnn(nn.Module):
    def __init__(self, input_size=200, hidden_size=64, dropout=0.5, num_classes=2, **kwargs):
        super(SimpleCnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.conv1d1 = nn.Conv1d(
            in_channels=3, out_channels=hidden_size, kernel_size=1, padding=0)

        self.conv1d2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv1d3 = nn.Conv1d(hidden_size, int(
            hidden_size * 2), kernel_size=3, padding=1)

        self.drop = nn.Dropout(p=self.dropout, inplace=False)

        self.layerNorm = nn.BatchNorm1d(int(hidden_size * 2))

        self.fc = nn.Linear(in_features=int(hidden_size * input_size),
                            out_features=int((hidden_size / 2)))

        self.fc2 = nn.Linear(in_features=int((hidden_size / 2)),
                             out_features=int(num_classes))

    def forward(self, x1, x2, length=None, hx=None):
        output = functional.relu(self.conv1d1(x1))
        output = functional.relu(self.conv1d2(output))
        output = functional.relu(self.conv1d2(output))
        output = self.pool2(functional.relu(self.conv1d2(output)))
        output = self.layerNorm(
            self.drop(functional.relu(self.conv1d3(output))))
        output = torch.flatten(output, 1)
        output = functional.relu(self.fc(output))
        return functional.softmax(self.fc2(output), dim=-1)


class CnnLinear(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, dropout=0.5, num_classes=2, **kwargs):
        super(CnnLinear, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.conv1d1 = nn.Conv1d(
            in_channels=3, out_channels=hidden_size, kernel_size=1, padding=0)

        self.conv1d2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv1d3 = nn.Conv1d(hidden_size, int(
            hidden_size * 2), kernel_size=3, padding=1)

        self.drop = nn.Dropout(p=self.dropout, inplace=False)

        self.layerNorm = nn.BatchNorm1d(int(hidden_size * 2))

        self.fc_cnn1 = nn.Linear(in_features=int(hidden_size * input_size),
                                 out_features=int((hidden_size)))

        self.metaNorm = nn.BatchNorm1d(linear_size)

        self.fc_meta1 = nn.Linear(
            in_features=linear_size, out_features=int(hidden_size / 4), bias=False)

        self.fc_meta2 = nn.Linear(in_features=int(
            hidden_size / 4), out_features=int(hidden_size / 4))

        self.fc_meta3 = nn.Linear(in_features=int(
            hidden_size / 4), out_features=int(hidden_size / 2))

        self.fc_meta4 = nn.Linear(in_features=int(
            hidden_size / 2), out_features=int(hidden_size / 2))

        self.fc_meta5 = nn.Linear(in_features=int(
            hidden_size / 2), out_features=hidden_size)

        self.fc_combine = nn.Linear(in_features=int(
            hidden_size * 2), out_features=int(hidden_size / 2))

        self.fc_combine2 = nn.Linear(in_features=int(
            (hidden_size / 2)), out_features=int(num_classes))

    def forward_cnn(self, x):
        output = functional.relu(self.conv1d1(x))
        output = functional.relu(self.conv1d2(output))
        # output = functional.relu(self.conv1d2(output))
        output = self.pool2(functional.relu(self.conv1d2(output)))
        # output = self.layerNorm(self.drop(functional.relu(self.conv1d3(output))))
        output = functional.relu(self.conv1d3(output))

        output = torch.flatten(output, 1)

        return functional.relu(self.fc_cnn1(output))

    def forward_meta(self, x):
        output = functional.relu(self.fc_meta1(x))
        output = functional.relu(self.fc_meta2(output))
        output = functional.relu(self.fc_meta3(output))
        output = functional.relu(self.fc_meta4(output))
        return functional.relu(self.fc_meta5(output))

    def forward(self, x1, x2):
        input_seq = x1
        input_meta = x2

        cnn_branch = self.forward_cnn(input_seq)
        # return functional.softmax(cnn_branch, dim=-1)
        meta_branch = self.forward_meta(input_meta)

        combine = torch.cat((cnn_branch, meta_branch), 1)
        combine = self.fc_combine(combine)

        return functional.softmax(self.fc_combine2(combine), dim=-1)


class LSTM_Classifier(nn.Module):
    def __init__(self, num_classes=2, input_size=200, hidden_size=64, num_layers=10, seq_length=200, dropout=0.5, batch_size=512):
        super(LSTM_Classifier, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.dropout = dropout  # dropout
        self.batch_size = batch_size  # batch size

        self.lstm = nn.LSTM(input_size=self.seq_length, hidden_size=hidden_size,
                            num_layers=self.num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(self.hidden_size, int(
            self.hidden_size / 2))  # fully connected 1
        # fully connected last layer
        self.fc = nn.Linear(int(self.hidden_size / 2), num_classes)

        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.split(x1, 1, dim=1)[0]
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # internal state
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        # Propagate input through LSTM
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # reshaping the data for Dense layer next
        output = output.view(-1, self.hidden_size)
        out = self.relu(output)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return functional.softmax(out, dim=-1)


class ConvLSTM_Classifier(nn.Module):
    def __init__(self, num_classes=2, input_size=200, hidden_size=64, num_layers=10, seq_length=200, dropout=0.5, batch_size=512):
        super(ConvLSTM_Classifier, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.dropout = dropout  # dropout
        self.batch_size = batch_size  # batch size

        self.lstm = nn.LSTM(input_size=self.seq_length, hidden_size=hidden_size,
                            num_layers=self.num_layers, batch_first=True)  # lstm
        self.conv1d1 = nn.Conv1d(
            in_channels=3, out_channels=1, kernel_size=1, padding=0)
        self.fc_1 = nn.Linear(self.hidden_size, int(
            self.hidden_size / 2))  # fully connected 1
        # fully connected last layer
        self.fc = nn.Linear(int(self.hidden_size / 2), num_classes)

        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = self.conv1d1(x1)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # internal state
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        # Propagate input through LSTM
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # reshaping the data for Dense layer next
        output = output.view(-1, self.hidden_size)
        out = self.relu(output)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return functional.softmax(out, dim=-1)


class AttentionNetwork(nn.Module):
    def __init__(self, input_size=200, hidden_size=64, k_max=20, dropout=0.5, num_classes=2, hidden_attention=128):
        super(AttentionNetwork, self).__init__()
        # define the dimensions of the FFNN
        self.input_size = input_size  # 1000
        self.hidden_size = hidden_attention  # 128
        self.num_classes = num_classes  # final output dimension
        self.k_max = k_max  # 20
        self.hidden_CNN = hidden_size  # 128
        self.dim = 2
        self.dropout = dropout  # dropout

        # transform the data using a CNN
        self.transformer_part1 = nn.Sequential(
            nn.Conv1d(3, self.hidden_CNN, kernel_size=11, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.transformer_part2 = nn.Sequential(
            # match the first numer with the output of cnn after pooling 20*
            nn.Linear(self.k_max * self.hidden_CNN, self.input_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.hidden_size, self.num_classes)

        # estiamte the attention weights
        self.attention = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        # the actual classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size * self.num_classes, 1),
            nn.Sigmoid()
        )

    def kmax_pooling(self, x):
        index = x.topk(self.k_max, dim=self.dim)[1].sort(dim=self.dim)[0]
        return x.gather(self.dim, index)

    def forward(self, x1, x2):
        # assumes input of form gene_count x channels x sequnce_length
        # example: 20x4x2000
        H = self.transformer_part1(x1)
        H = self.kmax_pooling(H)  # genes x kernels x max_feat

        # flatten out
        H = H.reshape(H.size(0), -1)
        # pass to the linear tranforamtion
        H = self.transformer_part2(H)

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)

        A = self.attention_weights(A_V * A_U)  # element wise multiplication

     #   A = self.attention(H)
        # A = torch.transpose(A, 1,0)
        A = functional.softmax(A, dim=1)
        return A


if __name__ == '__main__':

    x = torch.load('x1.pt')
    # model = LSTM_Classifier()
    # model = ConvLSTM_Classifier()
    # model = SimpleCnn()
    # model = AttentionNetwork()
    model = nucSimpleCnn()

    if torch.cuda.is_available():
        model.cuda()
        x.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    out = model.forward(x, x)
    print(out.shape)
