import os
import pickle

from cmdstanpy import CmdStanModel, from_csv
import numpy as np


def estimate_mvn_param(
        data_dict, 
        fname=None, 
        chains=3,
        n_warmup=500, 
        n_sampling=1000
    ):
    stan_file = os.path.join(".", "learn_underlying_normal.stan")
    model = CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=data_dict, 
        chains=chains, 
        iter_warmup=n_warmup, 
        iter_sampling=n_sampling, 
        show_console=False
    )

    if fname:
        fit.save_csvfiles('cifar_fits/'+fname)

    return fit

def get_mu_Sigma(fit):
    mu_estimate = np.zeros(n)
    Sigma_estimate = np.zeros((n,n))
    for i in range(n):
        mu_estimate[i] = fit.stan_variable("mu")[:,i].mean()
        for j in range(n):
            Sigma_estimate[i][j] = fit.stan_variable("Sigma")[:,i,j].mean()

    return mu, Sigma  


def get_consensus(arr):
    y = list(arr)
    most = max(list(map(y.count, y)))
    modes = list(set(filter(lambda x: y.count(x) == most, y)))
    if len(modes) > 1:
        return None 
    return modes[0]


def main():
    rerun_model = True

    with open('cifar_10h.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    K = data_dict["K"]

    chains=3
    n_warmup=1500
    n_sampling=2000

    n_items = 500
    mini_dict = data_dict.copy()
    mini_dict['Y_M'] = mini_dict['Y_M'][:n_items]
    mini_dict['Y_H'] = mini_dict['Y_H'][:n_items]
    mini_dict['n_items'] = n_items
    mini_dict['eta'] = 1

    fname = "fit_{}_{}_{}".format(n_warmup, n_sampling, n_items)

    n = (data_dict['n_models'] + data_dict['n_humans'])*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_mvn_param(mini_dict, fname, chains, n_warmup, n_sampling)
    else:
        fit = from_csv(path='cifar_fits/'+fname, method='sample')

        


if __name__ == "__main__":
    main()