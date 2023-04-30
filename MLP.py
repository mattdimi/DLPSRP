# Initial linear and log regression for baseline comparison

import numpy
import torch 

from MyDataLoader import *
import Model
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

class MLP(torch.nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        # Note: add error warnings for invalid dimensions 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU())
        for i in range(num_layers-3):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())


    def forward(self, x):
        # from class example
        return self.layers(x.reshape(-1, self.input_dim))
    
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
                prediction = self.layers(data.to(torch.float32))
                # print(label.dtype)
                # print(prediction.dtype)
                print(prediction.size())
                print(label.size())
                loss = self.loss_fn(prediction.flatten(), label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for batch in validset:
                x, y = batch
                y_hat = self.layers(x.to(torch.float32))
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
                y_hat = self.layers(x.to(torch.float32))
                predictions.append(y_hat)
            print(torch.cat(predictions).numpy())
            return torch.cat(predictions).numpy().flatten() # (n,) np.array

        # return self.forward(self, x)