import numpy
import torch 

from MyDataLoader import *
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

class LTSM(torch.nn.Module):
    def __init__(self, in_dimensions, hidden_dimensions, num_layers):
        super().__init__()
        #initialize parameters and model
        self.in_dimensions = in_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.num_layers = num_layers
        self.model = torch.nn.LSTM(in_dimensions, hidden_dimensions, num_layers, bias=True, device=None, dtype=None)
        #input = torch.rand(time_step, batch, in_dimensions)
        #c_0 = np.tanh(x*W_xc+H*W_hc+b_c)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input):
        output, hn, cn = self.model(input)
        return output, hn, cn
    
    def fit(self, trainset, validset, num_epochs = 10, max_patience = 3, min_delta = 0.001):
        # train loader?
        # trainloader = DataLoader(trainset)
        self.train()
        prev_val_loss = np.inf
        patience = 0
        for i in range(num_epochs):
            curr_val_loss = 0
            for batch in trainset: 
                data, label = batch
                # print(data.dtype)
                # data = data.to(torch.long)
                prediction = self.model(data.to(torch.float32))
                # print(label.dtype)
                # print(prediction.dtype)
                loss = self.loss_fn(prediction.flatten(), label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for batch in validset:
                x, y = batch
                y_hat = self.model(x.to(torch.float32))
                # x = x.to(torch.long)
                val_loss = self.loss_fn(y_hat.flatten(), y)
                curr_val_loss += val_loss
            if abs(curr_val_loss - prev_val_loss) < min_delta:
                patience+=1
            else:
                patience = 0
            if patience >= max_patience:
                break
        return self


    def predict(self, dataloader): # testing/projecting model 
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                x, _ = batch
                y_hat = self.model(x.to(torch.float32))
                predictions.append(y_hat)
            return torch.cat(predictions).numpy().flatten() # (n,) np.array