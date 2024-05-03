import os
import pickle
import logging

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
import pandas as pd

from utils import get_consensus, print_Sigma
from dataset import TestDataset
from consensus_model import ConsensusModel


def estimate_mvn_param(data_dict, chains, n_warmup, n_sampling):
    stan_file = os.path.join(".", "learn_underlying_normal.stan")
    model = CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=data_dict, 
        chains=chains, 
        iter_warmup=n_warmup, 
        iter_sampling=n_sampling, 
        show_console=False
    )
    return fit

def get_parameters(fit, n, interval=None):
    df = fit.draws_pd()

    mu = []
    for i in range(1,n+1):
        m = np.mean(df['mu[{}]'.format(i)])
        mu.append(m)
        print(round(m,3))
    
    print_Sigma(df, n)

    if interval:
        print("{}% interval: lower bound".format(interval*100))
        print_Sigma(df, n, 1-interval)
        print("{}% interval: upper bound".format(interval*100))
        print_Sigma(df, n, interval)

    L_Sigma = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            if j>=i:
                l_corr = np.mean(df['L_Sigma[{},{}]'.format(i,j)])
                L_Sigma[i-1][j-1] = l_corr
                L_Sigma[j-1][i-1] = l_corr

    return mu, L_Sigma

def query_n_random(dataset, n, i):
    random_candidate_experts = [i for i in range(dataset.n_humans)]
    model_prediction = dataset.get_model_prediction(i)
    votes = []
    for _ in range(n):
        random_expert = np.random.choice(random_candidate_experts)
        random_candidate_experts = [
            c for c in random_candidate_experts if c != random_expert
        ]
        votes.append(dataset.Y_H_new[i][random_expert])

    _, modes = get_consensus(votes)
    pred_y = np.random.choice(modes)
    return pred_y


def main():

    dataset_name = "nih" 
    # dataset_name = "cifar"

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    rerun_model = False

    n_items = 300
    chains = 3
    n_warmup= 1500
    n_sampling= 2000

    uncertainty_threshold = 0.1
    eta = 0.75

    with open('data/{}.pickle'.format(dataset_name), 'rb') as handle:
        data_dict = pickle.load(handle)

    dataset = TestDataset(
        n_models = data_dict['n_models'],
        n_humans = data_dict['n_humans'],
        n_classes = data_dict['K'],
        n_initialization_items = n_items,
        model_predictions = data_dict['Y_M'],
        human_predictions = data_dict['Y_H']
    )

    fname = "fit_{}_{}_{}".format(n_warmup, n_sampling, n_items)
    save_dir = dataset_name + '_fits'
    results_save_dir = dataset_name + '_results/'
    model_id = dataset_name + '_' + fname + '_' + str(uncertainty_threshold)

    if rerun_model:
        fit = estimate_mvn_param(
            dataset.get_init_stan_dict(eta=eta), 
            chains, 
            n_warmup, 
            n_sampling
        )
        fit.save_csvfiles(save_dir+'/'+fname)
    else:
        fit = from_csv(path=save_dir+'/'+fname, method='sample')

    mu, L_Sigma = get_parameters(fit, dataset.n)

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    stan_model = CmdStanModel(stan_file=stan_file_inf)
    consensus_model = ConsensusModel(dataset, fit, stan_model, model_id)

    n_tests = 50
    n_random_queries = 2
    results = []
    random_results = []

    for i in range(n_tests):

        print('running test on row ' + str(i))

        true_consensus = dataset.get_human_consensus(i)
        pred_y, n_queries = consensus_model.get_prediction(
            i, 
            uncertainty_threshold
        )
        random_pred_y = query_n_random(dataset, n_random_queries, i)

        result = 1 if pred_y == true_consensus else 0
        random_result = 1 if random_pred_y == true_consensus else 0

        results.append([n_queries, result])
        random_results.append([n_random_queries, random_result])

        print('human labels: ' + str(dataset.Y_H_new[i]))
        print("chosen:", pred_y)
        print("random/model prediction:", random_pred_y)
        print("actual:", true_consensus)

        unc_str = str(uncertainty_threshold)

        random_df = pd.DataFrame(random_results)
        random_df.to_csv(results_save_dir+"random_results" + str(n_random_queries) + ".csv")
        df = pd.DataFrame(results)
        df.to_csv(results_save_dir+"results" + unc_str + ".csv")

    print("completed testing")

if __name__ == "__main__":
    main()