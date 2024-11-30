from nll import *

import torch
from torch import nn
import numpy as np

class Custom_Layer(nn.Module):
    def __init__(self, in_features, out_features,
                 act, use_normalization, use_skip_connection) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = act

        if use_normalization:
            self.norm_layer = nn.LayerNorm([out_features])
        else:
            self.norm_layer = nn.Identity()
        
        if use_skip_connection:
            self.linear_2 = nn.Linear(out_features, out_features, bias = False)
            self.skip_connection_layer = nn.Linear(in_features, out_features)
            self.skip_connection = self.skip_connection_function
        else: 
            self.skip_connection = self.identity_for_pairs
            
    def skip_connection_function(self, x, res):
        return self.linear_2(res) + self.skip_connection_layer(x)

    def identity_for_pairs(self, x, res):
        return res

    def forward(self, x):
        res = self.linear_1(x)
        res = self.norm_layer(res)
        res = self.act(res)
        res = self.skip_connection(x, res)
        return res

class LatentModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.epoch = 0
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value) 
        
        # if kl_min = kl_max = 1, the loss will become the usual elbo
        self.betas = self.lambda_2 * np.concatenate((np.linspace(self.kl_min, self.kl_max, 12), np.ones(13)))

        if self.dtype == 'MNIST':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_bernoulli
            self.para = None  
        
        elif self.dtype == 'VerbAgg':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_bernoulli_reg
            # fixed effect
            self.beta = nn.Linear(self.fix_dim, 1, bias=self.fix_bias)
            # self.para will be passed into nll_bernoulli_reg
            self.para = [self.fix_dim, self.rand_dim, self.beta]
            assert self.fix_dim + self.rand_dim + 1 == self.x_dim, "Must be fix_dim + rand_dim + 1 == x_dim!"    

        elif self.dtype == 'copula':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_poisson_reg
            # fixed effect
            self.beta = nn.Linear(self.fix_dim, 1, bias=self.fix_bias)
            # self.para will be passed into nll_poisson
            self.para = [self.fix_dim, self.rand_dim, self.beta]
            assert self.fix_dim + self.rand_dim + 1 == self.x_dim, "Must be fix_dim + rand_dim + 1 == x_dim!"    
        
        elif self.dtype == 'cortex':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_nb
            # inverse dispersion for NB
            self.para = torch.nn.parameter.Parameter(torch.zeros(1, 1, self.u_dim))
        
        else:
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_normal
            # Note that if u ~ N(0, sigma2) and x = u + N(0, 1), then x = N(0, 1+sigma2) and std(x) = sqrt(1+sigma2)
            # Normalizing x, we have x / std(x) = N(0, 1) = u / std(x) + N(0, 1/var(x))
            # So, the sigma2 for likelihood is 1/var(x), not 1. After training, we obtain a sample of U by decoder(z) * std(x)
            self.para = torch.Tensor(0.5 / self.sigma2).to(self.device)

    def VAE_forward(self, x):
        ## forward
        # x = [batch_size, 1, x_dim]
        # z = [batch_size, num_samples, z_dim]
        # u = [batch_size, num_samples, u_dim]
        mu, logvar = self.encoder(x)
        mu = mu.expand(-1, self.num_samples, -1)
        logvar = logvar.expand(-1, self.num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        u = self.decoder(z)

        # loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2))
        ll_loss = torch.mean(self.cal_nll(x, u, self.para))

        return ll_loss, kl_loss, ll_loss + self.betas[self.epoch % 25] * kl_loss
    
    def IWAE_forward(self, x):
        # We refered to https://github.com/andrecavalcante/iwae/blob/main/iwae.py

        # computing mean and std of Gaussian proposal q(h|x)
        q_mean, q_logvar = self.encoder(x)
        q_std = torch.exp(q_logvar / 2)

        # replicating mean and std to generate multiple samples. Unsqueezing to handle batch sizes bigger than 1.
        # q_mean = torch.repeat_interleave(q_mean.unsqueeze(1), self.num_samples, dim=1)
        # q_std = torch.repeat_interleave(q_std.unsqueeze(1), self.num_samples, dim=1)
        q_mean = torch.repeat_interleave(q_mean, self.num_samples, dim=1)
        q_std = torch.repeat_interleave(q_std, self.num_samples, dim=1)

        # generating proposal samples
        # size of h: (batch_size, num_samples, h_size)
        h = q_mean + q_std * torch.randn_like(q_std)
        
        # computing mean of a Bernoulli likelihood p(x|h)
        likelihood_mean = self.decoder(h)

        # log p(x|h)
        # x = x.unsqueeze(1) # unsqueeze for broadcast
        log_px_given_h = -self.cal_nll(x, likelihood_mean, self.para) # sum over x_dim

        # gaussian prior p(h)
        log_ph = torch.sum(-0.5* torch.log(torch.tensor(2*np.pi)) - torch.pow(0.5*h,2), dim=-1) # sum over h_dim

        # evaluation of a gaussian proposal q(h|x)
        log_qh_given_x = torch.sum(-0.5* torch.log(torch.tensor(2*np.pi))-torch.log(q_std) - 0.5*torch.pow((h-q_mean)/q_std, 2), dim=-1) # sum over h_dim
        
        # computing log weights 
        log_w = log_px_given_h + log_ph - log_qh_given_x
       
        # normalized weights through Exp-Normalization trick
        M = torch.max(log_w, dim=-1)[0].unsqueeze(1)
        normalized_w =  torch.exp(log_w - M)/ torch.sum(torch.exp(log_w - M), dim=-1).unsqueeze(1) # unsqueeze for broadcast

        # loss signal        
        loss = torch.sum(normalized_w.detach().data * (log_px_given_h + log_ph - log_qh_given_x), dim=-1) # sum over num_samples
        loss = -torch.mean(loss) # mean over batchs

        # computing log likelihood through Log-Sum-Exp trick
        # log_px = M + torch.log((1/self.num_samples)*torch.sum(torch.exp(log_w - M), dim=-1))  # sum over num_samples
        # log_px = torch.mean(log_px) # mean over batches

        return None, None, loss

    def IS_forward(self, x):
        ## forward
        # x = [batch_size, 1, x_dim]
        # z = [batch_size, num_samples, z_dim]
        # u = [batch_size, num_samples, u_dim]
        mu, logvar = self.encoder(x)
        mu = mu.expand(-1, self.num_samples, -1)
        logvar = logvar.expand(-1, self.num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std 
        qz_values = torch.exp(-0.5 * self.lambda_1 * torch.sum(eps.pow(2), dim=2))

        u = self.decoder(z)

        nll = self.cal_nll(x, u, self.para)
        likelihood = torch.exp(-self.lambda_1 * nll.to(torch.float64)).detach()
        prior = torch.exp(-0.5 * self.lambda_1 * torch.sum(z.pow(2), dim=2))
        pz_values = prior * likelihood

        weights = pz_values / qz_values
        weights_sum = torch.sum(weights, dim=1)
        weights_sum_mask = (weights_sum == 0)
        weights[weights_sum_mask] = 1/self.num_samples
        IS_loss = torch.sum(weights * nll, dim=1) / torch.sum(weights, dim=1)
        
        # loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2))
        ll_loss = torch.mean(IS_loss)

        return ll_loss, kl_loss, ll_loss + self.betas[self.epoch % 25] * kl_loss
    
    def IS_cortex_forward(self, x):
        ## forward
        # x = [batch_size, 1, x_dim]
        # z = [batch_size, num_samples, z_dim]
        # u = [batch_size, num_samples, u_dim]
        mu, logvar = self.encoder(x)
        mu = mu.expand(-1, self.num_samples, -1)
        logvar = logvar.expand(-1, self.num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        qz_values = torch.exp(-0.5 * self.lambda_1 * torch.sum(eps.pow(2), dim=2))
        u = self.decoder(z)
        u = u * torch.sum(x, dim=2, keepdim=True)
        u = u.clamp(1e-10)        

        nll = self.cal_nll(x, u, self.para)
        likelihood = torch.exp(-self.lambda_1 * nll.to(torch.float64)).detach()
        prior = torch.exp(-0.5 * self.lambda_1 * torch.sum(z.pow(2), dim=2))
        pz_values = prior * likelihood

        weights = pz_values / qz_values
        # weights_sum = torch.sum(weights, dim=1)
        # weights_sum_mask = (weights_sum == 0)
        # weights[weights_sum_mask] = 1/self.num_samples
        IS_loss = torch.sum(weights * nll, dim=1) / torch.sum(weights, dim=1)
        
        # loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2))
        ll_loss = torch.mean(IS_loss)

        return ll_loss, kl_loss, ll_loss + self.betas[self.epoch % 25] * kl_loss

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.h_dims = [self.x_dim] + list(reversed(self.h_dims))

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.act, self.use_normalization, self.use_skip_connection_encoder))
        
        if self.use_skip_connection_encoder:
            self.mu = Custom_Layer(self.h_dims[-1], self.z_dim, self.act, self.use_normalization, self.use_skip_connection_encoder)
            self.logvar = Custom_Layer(self.h_dims[-1], self.z_dim, self.act, self.use_normalization, self.use_skip_connection_encoder)
        else:
            self.mu = nn.Linear(self.h_dims[-1], self.z_dim)
            self.logvar = nn.Linear(self.h_dims[-1], self.z_dim)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, 1, x_dim]
        # mu, logvar : [batch_size, 1, z_dim]
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -20, 20)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)   
        
        self.h_dims = [self.z_dim] + self.h_dims

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.act, self.use_normalization, self.use_skip_connection_decoder))
        
        if self.use_skip_connection_decoder:
            layers.append(Custom_Layer(self.h_dims[-1], self.u_dim, self.act, self.use_normalization, self.use_skip_connection_decoder))
        else:
            layers.append(nn.Linear(self.h_dims[-1], self.u_dim))
        
        if self.dtype == 'cortex':
            layers.append(nn.Softmax(dim=2))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        # z : [batch_size, num_samples, z_dim]
        # u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        return u