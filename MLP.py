import numpy as np
import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, model_type = "regression"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        output_dim = 1 if model_type == "regression" else 2
        self.model_type = model_type
        self.loss = torch.nn.MSELoss() if model_type == "regression" else torch.nn.CrossEntropyLoss()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            torch.nn.ReLU()
            )
        for i in range(num_layers-3):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float64))
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.layers(x)
    
    def fit(self, train_dataloader, valid_dataloader, num_epochs = 10, max_patience = 3, min_delta = 0.001):
        self.train()
        prev_val_loss = np.inf
        patience = 0
        for i in range(num_epochs):
            curr_val_loss = 0
            for batch in train_dataloader:
                x, y = batch
                y = y if self.model_type == "regression" else y.to(torch.long) 
                prediction = self(x)
                loss = self.loss(prediction, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for batch in valid_dataloader:
                x, y = batch
                y = y if self.model_type == "regression" else y.to(torch.long)
                y_hat = self(x)
                val_loss = self.loss(y_hat, y)
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
                y_hat = self(x) if self.model_type == "regression" else self(x).argmax(dim=1)
                predictions.append(y_hat)
            return torch.cat(predictions).numpy().flatten() # (n,) np.array