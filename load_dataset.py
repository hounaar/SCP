import torch
import os
from torch.utils.data import Dataset, DataLoader

# ✅ Move `CustomDataset` outside
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

def load_dataset(batch_size: int, path="./data"):
    X_train = torch.load(os.path.join(path, "X_train.pt"))
    X_test = torch.load(os.path.join(path, "X_test.pt"))
    Y_train = torch.load(os.path.join(path, "Y_train.pt"))
    Y_test = torch.load(os.path.join(path, "Y_test.pt"))

    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
