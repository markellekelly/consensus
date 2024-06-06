import os
import pickle
import logging
import shutil

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
import pandas as pd

from utils import get_consensus, print_Sigma
from dataset import TestDataset
from consensus_model import ConsensusModel


def estimate_mvn_param(data_dict, chains, n_warmup, n_sampling, id_str):
    stan_file = os.path.join(".", "update_parameters.stan")
    model = CmdStanModel(stan_file=stan_file)
    out_dir = "tmp/" + "mvn_param" + id_str
    fit = model.sample(
        data=data_dict, 
        chains=chains, 
        output_dir=out_dir,
        iter_warmup=n_warmup, 
        iter_sampling=n_sampling, 
        show_console=False
    )
    return fit, out_dir


def main():

    # options:  "nih", "nih_noisy_model", "cifar", "cifar_alt46", "cifar_alt28"
    #           "imagenet_m1", "imagenet_m2"
    dataset_name = "cifar"

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    chains = 3
    n_warmup= 1500
    n_sampling= 2000
    unique_id = ''
    
    eta = 0.75
    use_temp_scaling = 1

    fname = "fit_{}_{}_{}_{}".format(n_warmup, n_sampling, eta)
    results_save_dir = dataset_name + '_results/'
    model_id = dataset_name + '_' + fname + "_" + unique_id

    with open('data/{}.pickle'.format(dataset_name), 'rb') as handle:
        data_dict = pickle.load(handle)

    results = []

    thresholds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    for threshold in thresholds:

        n_tests = 250
        start_point = 1
        Y_H_full = [data_dict['Y_H'][0]]
        Y_M_full = [data_dict['Y_M'][0]]
        i = 1
        out_dir = None

        for t in range(start_point, start_point+n_tests):
            # update after every data point for the first 20 iterations, 
            # then every 10 until 100, then every 50
            if t < 20 or (t % 10 == 0 and t < 100) or (t % 50 == 0 and t < 250):
                dataset = TestDataset(
                    n_models = data_dict['n_models'],
                    n_humans = data_dict['n_humans'],
                    n_classes = data_dict['K'],
                    n_initialization_items = t,
                    model_predictions = Y_M_full,
                    human_predictions = Y_H_full,
                    model_predictions_test = data_dict['Y_M'][t:],
                    human_predictions_test = data_dict['Y_H'][t:]
                )
                init_dict = dataset.get_init_stan_dict(eta=eta)
                fit_new, out_dir = estimate_mvn_param(
                        init_dict, 
                        chains, 
                        n_warmup, 
                        n_sampling,
                        id_str = str(t) + "_" + model_id
                )
                stan_file_inf = os.path.join(".", "simulate_consensus.stan")
                stan_model = CmdStanModel(stan_file=stan_file_inf)
                consensus_model = ConsensusModel(dataset, fit_new, stan_model, model_id+str(t))
                i = 0

            result, Y_O, Y_M = consensus_model.get_prediction(i, threshold)
            # result, Y_O, Y_M = consensus_model.get_prediction_random_querying(i, threshold)
            result['threshold'] = threshold
            result['data_index'] = t
            results.append(result)
            Y_H_full.append(Y_O)
            Y_M_full.append(Y_M)
            i += 1

            df = pd.DataFrame(results)
            df.to_csv(results_save_dir+"results.csv")



if __name__ == "__main__":
    main()