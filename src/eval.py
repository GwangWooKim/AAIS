from util import *

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics

labelencoder = LabelEncoder()

def eval_(model, u, train_loader, save, title = None, save_path = None):

    if save_path == None:
        save_path = './test'
    
    if save:
        os.makedirs(save_path, exist_ok=True)

    fix_seed(model.seed)
    with torch.inference_mode():
        if model.dtype == 'MNIST':
            eval_MNIST(model, u, train_loader, save, title, save_path)

        elif model.dtype == 'VerbAgg':
            eval_VerbAgg(model, u, train_loader, save, title, save_path)

        elif model.dtype == 'copula':
            eval_copula(model, u, train_loader, save, title, save_path)

        elif model.dtype == 'cortex':
            eval_cortex(model, u, train_loader, save, title, save_path)
        
        elif model.dtype == 'circle':
            eval_circle(model, u, train_loader, save, title, save_path)

        elif model.dim == 1:
            eval_1d(model, u, train_loader, save, title, save_path)

def eval_MNIST(model, u, train_loader, save, title, save_path):
    num_generated = 50
    z = torch.randn(num_generated, model.z_dim).to(model.device)
    generated = torch.sigmoid(model.decoder(z)).detach().cpu().numpy()

    # 50 generated digits
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    for i in range(num_generated):
        subplot = fig.add_subplot(5, 10, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(generated[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.tight_layout()

    if save:
        plt.savefig(f'{save_path}/generated_images.png')
    else:
        plt.show()

def eval_post(x_i, w_i, y_i, model, u_post):
    # nll
    x_i = x_i.squeeze(1)
    w_i = w_i.squeeze(1)
    y_i = y_i.flatten()
    beta = model.beta.weight.data.detach().cpu()
    linear = (beta * x_i).sum(1) + (w_i * u_post).sum(1)
    prob = torch.sigmoid(linear)
    log_likelihood = y_i * torch.log(prob) + (1 - y_i) * torch.log(1 - prob)
    nll = -sum(log_likelihood).item()

    # sum of square of Pearson residuals
    p_resid = torch.sum(torch.square((y_i - prob) / torch.sqrt(prob * (1-prob)))).item()

    # sum of square of deviance residuals
    binomial_dist = torch.distributions.Binomial(total_count=1, probs=y_i)
    saturated_loglik = binomial_dist.log_prob(y_i)
    deviance = 2 * (saturated_loglik - log_likelihood)
    d_resid = torch.sum(torch.square(torch.sign(y_i - prob) * torch.sqrt(deviance))).item()
    return nll, p_resid, d_resid

def eval_VerbAgg(model, u, train_loader, save, title, save_path):
    x_end = model.fix_dim
    w_end = model.fix_dim + model.rand_dim
    y_end = model.fix_dim + model.rand_dim + 1

    data = train_loader.dataset
    x_i, w_i, y_i = data[:, :, :x_end], data[:, :, x_end:w_end], data[:, :, w_end:y_end]

    # make noise
    z = torch.randn(1, 10000, model.z_dim).to(model.device)
    # generate prior
    u_prior = model.decoder(z).detach().cpu().expand(train_loader.dataset.size(0), -1, -1)
    
    for i in range(5):
        plt.figure()
        plt.hist(u_prior[0, :, i], bins = 50)
        plt.tight_layout()
        if save:
            plt.savefig(f'{save_path}/prior_{i}.png')
        else:
            plt.show()
        
        for j in range(i+1, 5):
            plt.figure()
            plt.scatter(u_prior[0, :, i], u_prior[0, :, j])
            plt.tight_layout()

            if save:
                plt.savefig(f'{save_path}/prior_{i}{j}.png')
            else:
                plt.show()

    # estimate posterior
    beta = model.beta.weight.data.unsqueeze(0).detach().cpu()
    linear = torch.sum(beta * x_i, axis=-1, keepdims=True) + torch.sum(w_i * u_prior, axis=-1, keepdims=True)
    prob = torch.sigmoid(linear)
    likelihood = (prob**y_i * (1 - prob)**(1 - y_i))
    u_post = torch.sum(likelihood * u_prior, axis=1) / torch.sum(likelihood, axis=1)
    
    # save fixed effects
    fixed_effects = u_prior[0, :, :].mean(0)
    if save:
        torch.save(fixed_effects, f'{save_path}/fixed_effects.pt')
    else:
        print('fixef:', fixed_effects)
    
    # negative log-likelihood
    nll, p_resid, d_resid = eval_post(x_i, w_i, y_i, model, u_post)
    dict_ = {'nll' : nll, 'p_resid' : p_resid, 'd_resid' : d_resid}
    if save:
        torch.save(dict_, f'{save_path}/evaluation.pt')
    else: 
        print('eval:', dict_)

def true_lambda(x):
    # phi(t) / (dphi(t)/dt)
    return -0.5 * (x - x**3)

def lambda_estimate(u):
    # phi(t) / (dphi(t)/dt) can be estimated by Kendall’s measure, that is, 
    # phi(t) / (dphi(t)/dt) = t - K_{C}(t) (An Introduction to Copulas, Nelsen, Theorem 4.3.4)
    u1 = u[:, 0]
    u2 = u[:, 1]

    n = len(u1)
    z = np.zeros(n)

    for i in range(n):
        z[i] = np.mean((u1 <= u1[i]) & (u2 <= u2[i]))
    
    res = ss.ecdf(z)
    
    points = np.linspace(0, 1, 400)
    values = points - res.cdf.evaluate(points) # t - K_{C}(t)
    return values

def eval_copula(model, u, train_loader, save, title = None, save_path = None):
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()

    ## dimension-wise histograms
    for i in range(model.u_dim):

        # quantative result
        ks, wd = cal_distances(generated[:, i], u[:, i], model.dtype)
        
        plt.figure()
        max_value = 1.1 * max(plt.hist(u[:, i], bins=100, label='True', alpha = 0.4, density=True, color='C1')[0])
        plt.ylim(0, max_value)
        plt.hist(generated[:, i], bins=100, label='Estimated', alpha = 0.4, density=True, color='C2')
        plt.text(0.99,  0.99, f'KS = {ks} \nWD = {wd}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(f'{save_path}/histgram_{i+1}.png')
        else:
            plt.show()
    
    ## Lambda function estimation
    points = np.linspace(0, 1, 400)
    true = true_lambda(points)
    estimated = lambda_estimate(generated)
    L1 = round((np.abs(true - estimated)/400).sum(), 3)
    L2 = round((np.square(true - estimated)/400).sum(), 3)

    plt.figure()
    plt.plot(points, true, label='True', color='C1')
    plt.plot(points, estimated, label='Estimated', color='C2')
    plt.legend()
    plt.text(0.99,  0.05, f'L1 distance = {L1}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.text(0.99,  0.01, f'L2 distance = {L2}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/lambda_function.png')
    else:
        plt.show()

    # for save distances
    distances = {'ks': [cal_distances(generated[:, i], u[:, i], model.dtype)[0] for i in range(model.z_dim)],
                'wd': [cal_distances(generated[:, i], u[:, i], model.dtype)[1] for i in range(model.z_dim)]}
    distances['L1'] = L1
    distances['L2'] = L2
    if save:
        torch.save(distances, f'{save_path}/dist.pt')

def eval_cortex(model, u, train_loader, save, title = None, save_path = None):
    mu, _ = model.encoder(train_loader.dataset.to(model.device))

    umap_2 = umap.UMAP(n_components=2, random_state=42)
    embedding = umap_2.fit_transform(mu.squeeze(1).detach().cpu().numpy())

    clustering = KMeans(random_state=model.seed, n_clusters=len(set(u))).fit(mu.squeeze(1).detach().cpu().numpy())
    ARI = metrics.adjusted_rand_score(u, clustering.labels_)
    NMI = metrics.normalized_mutual_info_score(u, clustering.labels_)

    dist = {'ARI' : ARI, 'NMI' : NMI}
    if save:
        with open(f'{save_path}/evaluation.json', 'w') as f: 
            json.dump(dist, f, indent=4)
    else:
        print('ARI:', ARI)
        print('NMI:', NMI)

    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
            c = labelencoder.fit_transform(u), cmap = 'Spectral', s=20,
            )
    legend = ax.legend(*(scatter.legend_elements()[0], labelencoder.classes_),
                        loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/visualization.png')
    else:
        plt.show()

def eval_circle(model, u, train_loader, save, title = None, save_path = None):

    # sample generation
    x = train_loader.dataset.squeeze(1).numpy()
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()

    # for 2d scatter plot
    plt.figure()
    plt.scatter(generated[:, 0], generated[:, 1], label = 'generated')
    plt.scatter(x[:, 0], x[:, 1], label = 'train data')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/scatterplot.png')
    else:
        plt.show()

    # for 2d hists
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    h1 = ax1.hist2d(x[:, 0], x[:, 1], bins=100, density=True)
    ax1.set_title('train data')
    h2 = ax2.hist2d(generated[:, 0], generated[:, 1], bins=100, density=True)
    ax2.set_title('generated')

    cbar1 = fig.colorbar(h1[3], ax=ax1)
    cbar1.set_label('Density')
    cbar2 = fig.colorbar(h2[3], ax=ax2)
    cbar2.set_label('Density')

    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/histogram.png')
    else:
        plt.show()

def eval_1d(model, u, train_loader, save, title = None, save_path = None):

    # sample generation
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()
    generated = generated * train_loader.std + train_loader.mean
    
    # quantative result
    ks, wd = cal_distances(generated.reshape(-1, ), u.reshape(-1, ), model.dtype)
    dist = {'ks' : ks, 'wd' : wd}

    # compact domain check
    if model.dtype == 'exp':
        fp = (generated < 0).sum() / len(generated)
        dist['fp'] = fp

    if save:
        with open(f'{save_path}/distances.json', 'w') as f: 
            json.dump(dist, f, indent=4)

    # histogram
    plt.figure()
    max_value = 1.1 * max(plt.hist(u, bins=100, label='True', alpha = 0.4, density=True, color='C1')[0])
    plt.ylim(0, max_value)
    plt.hist(generated, bins=100, label='Estimated', alpha = 0.4, density=True, color='C2')
    plt.text(0.99,  0.99, f'KS = {ks} \nWD = {wd}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/histogram.png')
    else:
        plt.show()
        

