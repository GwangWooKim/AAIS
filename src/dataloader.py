import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

def data_loader(dtype, dim, batch_size):

    std = None

    if dtype == 'MNIST':
        MNIST_train = torchvision.datasets.MNIST(root='./data', download=True, train=True)
        MNIST_test = torchvision.datasets.MNIST(root='./data', download=True, train=False)
        x, u = torch.round(MNIST_train.data / 255).view(-1, 28 * 28), torch.round(MNIST_test.data / 255).view(-1, 28 * 28)
        train_loader = DataLoader(x.unsqueeze(1), batch_size = batch_size, shuffle = True)

    elif dtype == 'VerbAgg':
        df = pd.read_csv('./data/VerbAgg.csv', index_col=0)
        x = torch.from_numpy(df.values)
        u = None
        train_loader = DataLoader(x.to(torch.float32).unsqueeze(1), batch_size = batch_size, shuffle = True)

    elif dtype == 'copula':
        copula = pd.read_csv('./data/copula.csv', index_col=0)
        x = copula.drop(['u.1', 'u.2', 'u.3', 'u.4', 'u.5'], axis=1).values
        u = copula[['u.1', 'u.2', 'u.3', 'u.4', 'u.5']].values
        train_loader = DataLoader(torch.from_numpy(x).unsqueeze(1).to(torch.float32), batch_size = batch_size, shuffle = True)

    elif dtype in ['cortex', 'circle']:
        x = np.load(f'./data/{dtype}_x.npy', allow_pickle=True)
        u = np.load(f'./data/{dtype}_u.npy', allow_pickle=True)
        train_loader = DataLoader(torch.from_numpy(x).unsqueeze(1).to(torch.float32), batch_size = batch_size, shuffle = True)
        
    else:
        x = np.load(f'./data/{dtype}1d_x.npy')
        u = np.load(f'./data/{dtype}1d_u.npy')

        mean = x.mean(0)
        std = x.std(0)
        x = (x - mean) / std
        train_loader = DataLoader(torch.from_numpy(x).unsqueeze(1).to(torch.float32), batch_size = batch_size, shuffle = True)
        train_loader.mean = mean
    
    train_loader.std = std
    
    return x, u, train_loader
    
