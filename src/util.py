import torch
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn
import scipy.stats as ss

def fix_seed(seed_value = 42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_value)

def Activation(activation): 
    if activation == 'PReLU':
        act = nn.PReLU()
    elif activation == 'Softplus':
        act = nn.Softplus()    
    else: 
        Exception("activation must be one of PReLU or Softplus!")
    return act 

def wd_groupping(model):
    lst_wd = []
    lst_wd_no = []
    for name, parm in model.named_parameters():
        if 'linear' in name and 'bias' not in name:
            lst_wd.append(parm)
        else:
            lst_wd_no.append(parm)

    params_wd = {'params' : lst_wd, 'weight_decay' : model.wd}

    if (model.use_skip_connection_encoder or model.use_skip_connection_decoder):
        # in this case we apply weight decay only to linear weights
        params_wd_no = {'params' : lst_wd_no, 'weight_decay' : 0}
    else: 
        params_wd_no = {'params' : lst_wd_no, 'weight_decay' : model.wd}

    return [params_wd, params_wd_no]

class MixtureOfGaussians(ss.rv_continuous):
    def _pdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).pdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).pdf(x)
        return res
    
    def _cdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).cdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).cdf(x)
        return res

def cal_distances(generated, u, dtype):
    if dtype == 'normal':
        true = ss.norm(loc=0, scale=0.5)

    if dtype == 'exp':
        true = ss.expon(scale=2)
    
    if dtype == 'mix':
        true = MixtureOfGaussians(a=-np.inf, b=np.inf)

    if dtype == 'copula':
        true = ss.gamma(a = 2, loc=-2, scale=1)
        
    ks = ss.kstest(generated, true.cdf)[0]
    wd = ss.wasserstein_distance(u_values = generated, v_values = u, 
                                 #u_weights = np.ones_like(generated), v_weights = true.pdf(u)
                                 )
    ks = round(ks, 3)
    wd = round(wd, 3)
    return ks, wd