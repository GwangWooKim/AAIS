from util import *
from eval import *

from tqdm import tqdm
import time
import os
import json
import matplotlib.pyplot as plt

def train(model, optimizer, train_loader):
    fix_seed(model.seed)

    lst = []
    lst_ll = []
    lst_kl = []

    # define forward
    if model.ftype == 'VAE':
        def model_forward(x):
            ll_loss, kl_loss, loss = model.VAE_forward(x)
            return ll_loss, kl_loss, loss
    
    if model.ftype == 'IWAE':
        def model_forward(x):
            _, _, loss = model.IWAE_forward(x)
            return torch.Tensor([0]), torch.Tensor([0]), loss
            
    if model.ftype == 'IS':
        if model.dtype == 'cortex':
            # this is because the model has a different forward architecture
            def model_forward(x):
                ll_loss, kl_loss, loss = model.IS_cortex_forward(x)
                return ll_loss, kl_loss, loss 
        else:
            def model_forward(x):
                ll_loss, kl_loss, loss = model.IS_forward(x)
                return ll_loss, kl_loss, loss 

    model.train()
    for epoch in tqdm(range(model.epochs), desc='Training Loop'):
        train_loss = 0
        train_ll_loss = 0
        train_kl_loss = 0
        model.epoch = epoch
        for step, x in enumerate(train_loader):

            x = x.to(model.device)

            optimizer.zero_grad()
            ll_loss, kl_loss, loss = model_forward(x)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ll_loss += ll_loss.item()
            train_kl_loss += kl_loss.item()

        lst.append(train_loss / (step + 1))
        lst_ll.append(train_ll_loss / (step + 1))
        lst_kl.append(train_kl_loss / (step + 1))
    
    model.lst = lst
    model.eval()
    
    plt.plot(lst, label = 'loss')
    if len(lst_ll)>0:
        plt.plot(lst_ll, label = 'll_loss')
        plt.plot(lst_kl, label = 'kl_loss')
    plt.legend()

    return model

def save_model(model, save_path = None):
    if save_path == None:
        save_path = './test'
    os.makedirs(save_path, exist_ok=True)

    # save state
    torch.save(model.state_dict(), f'{save_path}/model_states.pt')

    # save kwargs
    with open(f'{save_path}/model_kwargs.json', 'w') as f: 
        model.kwargs['device'] = str(model.kwargs['device'])
        model.kwargs['act'] = str(model.kwargs['act'])
        model.kwargs['sigma2'] = str(model.kwargs['sigma2'])
        json.dump(model.kwargs, f, indent=4)
