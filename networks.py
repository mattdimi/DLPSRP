import torch
import numpy as np

class Base(torch.nn.Module):

    def __init__(self):
        """
        Base layer for the neural networks used implementing the training and prediction methods
        """
        super().__init__()

    def fit(self, train_dataloader:torch.utils.data.DataLoader, validation_dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, model_type = "regression", num_epochs = 10, max_patience = 3, min_delta = 0.001):
        """Function to train the model

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set
            validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set
            optimizer (torch.optim.Optimizer): Optimizer used
            loss (torch.nn.MSELoss): Loss function used
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            max_patience (int, optional): Maximum number of epochs where loss does not decrease more than `min_delta` consecutively. Defaults to 3.
            min_delta (float, optional): Min loss difference below which to consider that loss flattens. Defaults to 0.001.

        Returns:
            torch.nn.Module: the fitted model
        """
        self.train()
        prev_val_loss = np.inf
        patience = 0

        loss_fn = torch.nn.functional.mse_loss if model_type == "regression" else torch.nn.functional.cross_entropy
        
        for _ in range(num_epochs):
            curr_val_loss = 0
            for batch in train_dataloader:
                x, y = batch
                y = y.long() if model_type == "classification" else y
                y_hat = self(x)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for batch in validation_dataloader:
                x, y = batch
                y = y.long() if model_type == "classification" else y
                y_hat = self(x)
                val_loss = loss_fn(y_hat, y)
                curr_val_loss+=val_loss
            if abs(curr_val_loss - prev_val_loss) < min_delta:
                patience+=1
            else:
                patience = 0
            if patience >= max_patience:
                break
        return self
    
    def predict(self, dataloader: torch.utils.data.DataLoader):
        """Predict method

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader on which to generate predictions

        Returns:
            np.array: (n, ) array of predictions where n is the number of input observations
        """
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                x, _ = batch
                y_hat = self(x)
                y_hat = y_hat.argmax(dim = 1) if self.model_type == "classification" else y_hat
                predictions.append(y_hat)
            return torch.cat(predictions).numpy().flatten() # (n,) np.array
    
class LinearRegression(Base):
    def __init__(self, input_dim, lr = 1e-3):
        """Linear regression model

        Args:
            input_dim (_type_): input dimension
            lr (_type_, optional): learning rate of the optimizer. Defaults to 1e-3.
        """
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, dtype = torch.float64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, train_dataloader, valid_dataloader):
        return super().fit(train_dataloader, valid_dataloader, self.optimizer)

class MLP(Base):
    def __init__(self, input_dim, hidden_dim, num_layers, model_type):
        super().__init__()
        self.input_dim = input_dim
        output_dim = 1 if model_type == "regression" else 2
        self.model_type = model_type
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            torch.nn.ReLU()
            )
        for i in range(num_layers-3):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float64))
        self.final_activation = torch.nn.Sigmoid() if model_type == "classification" else torch.nn.Identity()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.final_activation(self.layers(x))

    def fit(self, train_dataloader, valid_dataloader):
        return super().fit(train_dataloader, valid_dataloader, self.optimizer, self.model_type)

class LSTM(Base):
    def __init__(self, input_size, hidden_size, num_layers, model_type):
        """Long-Short Term memory neural network

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            num_layers (_type_): _description_
            model_type (_type_): _description_
        """
        super().__init__()
        self.in_dimensions = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.output_dim = 1 if self.model_type == "regression" else 2
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dtype = torch.float64)
        self.linear = torch.nn.Linear(hidden_size, self.output_dim, dtype = torch.float64)
        self.final_activation = torch.nn.Sigmoid() if model_type == "classification" else torch.nn.Identity()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype = torch.float64)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype = torch.float64)
        output, _ = self.lstm(x, (h0, c0))
        return self.final_activation(self.linear(output[:,-1,:])) # (batch_size, hidden_size)

    def fit(self, train_dataloader, valid_dataloader):
        return super().fit(train_dataloader, valid_dataloader, self.optimizer, self.model_type)

class CNN(Base):
    def __init__(self, num_layers, channels, num_classes, model_type):
        kernel = 1
        self.model = torch.nn.Sequential(torch.nn.Conv1d(1, channels[0]), torch.nn.MaxPool1D(kernel), torch.nn.ReLU())
        for i in range(len(channels)-1):
            self.model.append(torch.nn.Conv1d(channels[i], channels[i+1]))
            self.model.append(torch.nn.MaxPool1d(kernel))
            self.model.append(torch.nn.ReLU())
        self.out_layer = torch.nn.Linear(channels[len(channels)-1], num_classes)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.model_type = model_type

    def forward(self, input: torch.Tensor): # modified from cnn hw
        out = self.model(input)
        out = self.out_layer(torch.flatten(out, start_dim=1))
        return out

    def fit(self, train_dataloader, valid_dataloader):
        return super().fit(train_dataloader, valid_dataloader, self.optimizer, self.model_type)


class GRU(Base):
    def __init__(self, in_dimensions, hidden_dimensions, num_layers, model_type):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.num_layers = num_layers
        self.model = torch.nn.GRU(in_dimensions, hidden_dimensions, num_layers, bias=True, device=None, dtype=None)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.model_type = model_type

    def forward(self, input):
        output, hn = self.model(input)
        return output, hn
