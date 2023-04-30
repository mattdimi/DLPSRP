import numpy as np
import torch

class CNN(torch.nn.Module):
    def __init__(self, num_layers, channels, num_classes):
        kernel = 1
        self.model = torch.nn.Sequential(torch.nn.Conv1d(1, channels[0]), torch.nn.MaxPool1D(kernel), torch.nn.ReLU())
        for i in range(len(channels)-1):
            self.model.append(torch.nn.Conv1d(channels[i], channels[i+1]))
            self.model.append(torch.nn.MaxPool1d(kernel))
            self.model.append(torch.nn.ReLU())
        self.out_layer = torch.nn.Linear(channels[len(channels)-1], num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input: torch.Tensor): # modified from cnn hw
        out = self.model(input)
        out = self.out_layer(torch.flatten(out, start_dim=1))
        return out
    
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
