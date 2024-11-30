import argparse
import json
import subprocess

from dataloader import *
from model import *
from util import *
from train import *
from eval import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default = 42, type = int)
parser.add_argument("-d", "--dtype", choices = ['exp' , 'mix', 'copula', 'MNIST', 'cortex', 'VerbAgg'], default = 'exp')
args = parser.parse_args()


def main():
    with open(f'./config/model_kwargs_{args.dtype}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    # for data
    dtype = config['dtype']
    dim = config['dim']
    batch_size = config['batch_size']
    x, u, train_loader = data_loader(dtype, dim, batch_size)

    ## for model 
    # x_dim = full data dimension
    # h_dims = must be a list 
    # z_dim = latent dimension
    # u_dim = generated dimension
    # ftype = must be one of VAE or IS
    # num_samples = number of samples for Monte Carlo
    # activation = must be one of PReLU or Softplus
    # use_normlization = whether to use layer normalization
    # use_skip_connection = whether to use skip connection
    # kl_min = minimum value of kl_scheduler 
    # kl_max = maximum value of kl_scheduler
    x_dim = config['x_dim']
    h_dims = config['h_dims']
    z_dim = config['z_dim']
    u_dim = config['u_dim']
    ftype = config['ftype']
    num_samples = config['num_samples']
    activation = 'PReLU'
    use_normlization = config['use_normalization']
    use_skip_connection_encoder = config['use_skip_connection_encoder']
    use_skip_connection_decoder = config['use_skip_connection_decoder']
    kl_min = config['kl_min']
    kl_max = 1

    ## for model (necessary only for some models)
    # sigma2 = used for normal likelihood
    # fix_dim = covariate dimension for fixed effects
    # fix_bias = whether to add an intercept term
    # rand_dim = covariate dimension for random effects
    # lambda_1 = a constant for power function
    # lambda_2 = a weight for kl_loss only when lambda_1 is used
    sigma2 = 1 / (train_loader.std)**2 if train_loader.std is not None else np.array([0.01])
    fix_dim = 0 if dtype == 'VerbAgg' else 5
    fix_bias = False 
    rand_dim = u_dim
    lambda_1 = 1e-4 if dtype == 'cortex' and ftype == 'IS' else 1
    lambda_2 = 1e-4 if dtype == 'cortex' and ftype == 'IS' else 1

    # for train
    lr = config['lr']
    wd = config['wd']
    epochs = config['epochs']
    seed = args.seed
    name = subprocess.check_output(['hostname']).decode().replace('\n','')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fix_seed(seed)
    model = LatentModel(
        dtype = dtype, dim = dim, batch_size = batch_size, 
        x_dim = x_dim, h_dims = h_dims, z_dim = z_dim, u_dim = u_dim, 
        ftype = ftype, act = Activation(activation), num_samples = num_samples, 
        use_normalization = use_normlization, use_skip_connection_encoder = use_skip_connection_encoder, use_skip_connection_decoder = use_skip_connection_decoder,
        kl_min = kl_min, kl_max = kl_max,
        sigma2 = sigma2, fix_dim = fix_dim, fix_bias = fix_bias, rand_dim = rand_dim, lambda_1 = lambda_1, lambda_2 = lambda_2,
        lr = lr, wd = wd, epochs = epochs, seed = seed, name = name, device = device,
    ).to(device)

    # if skip_connection = False, it works like an usual weight decay
    optimizer = torch.optim.Adam(wd_groupping(model), lr = model.lr)
    model = train(model, optimizer, train_loader)
    eval_(model, u, train_loader, save = True, title = None, save_path = f'./{ftype}_{dtype}')
    save_model(model, save_path = f'./{ftype}_{dtype}')
    
if __name__ == '__main__':
    main()
