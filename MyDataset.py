import torch

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        """Dataset class to load data into a dataset

        Args:
            X (np.array): features of shape (number of samples, number of features)
            y (np.array): target of shape (number of samples, )
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]