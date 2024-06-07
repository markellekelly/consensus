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


def get_fnames(dataset_name, noisy, unique_id):
    results_save_dir = dataset_name + str(noisy) + '_results/'
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    model_id = dataset_name + str(noisy) + '_fit_' + unique_id
    nois_str = "" if noisy == 0 else "_noisy"
    if noisy == 2:
        nois_str += "2"
    data_file = 'data/{}/data{}.pickle'.format(dataset_name, nois_str)

    return data_file, results_save_dir, model_id

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

    # dataset options:  "nih", "cifar", "imagenet"
    dataset_name = "nih"
    # noisy options: 0, 1 (for cifar also 2)
    noisy = 0
    n_tests = 250

    chains = 3
    n_warmup= 1500
    n_sampling= 2000
    id_str = ''
    
    eta = 0.75
    use_temp_scaling = 1
    use_correlations = 1

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    data_file, results_save_dir, id_ = get_fnames(dataset_name, noisy, id_str)

    with open(data_file, 'rb') as handle:
        data_dict = pickle.load(handle)

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    results = []

    thresholds = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    thresholds = [0.05, 0.1, 0.3]
    start_point = 1
    for threshold in thresholds:

        Y_H_observed = [data_dict['Y_H'][0]]
        i = 1
        out_dir = None

        dataset = TestDataset(
            n_models = data_dict['n_models'],
            n_humans = data_dict['n_humans'],
            n_classes = data_dict['K'],
            model_predictions_test = data_dict['Y_M'],
            human_predictions_test = data_dict['Y_H'],
            use_temp_scaling = use_temp_scaling,
            eta = eta
        )

        for t in range(start_point, start_point+n_tests):
            # update after every data point for the first 20 iterations, 
            # then every 10 until 100, then every 50
            if t < 20 or (t % 10 == 0 and t < 100) or (t % 50 == 0 and t < 250):
                dataset.update(i, Y_H_observed)
                init_dict = dataset.get_init_stan_dict()
                fit_new, out_dir = estimate_mvn_param(
                        init_dict, 
                        chains, 
                        n_warmup, 
                        n_sampling,
                        id_str = str(t) + "_" + id_
                )
                stan_model = CmdStanModel(stan_file=stan_file_inf)
                consensus_model = ConsensusModel(
                    dataset, 
                    fit_new, 
                    stan_model, 
                    id_+str(t)
                )
                i = 0; Y_H_observed = []

            result, Y_O, Y_M = consensus_model.get_prediction(i, threshold)
            # result, Y_O, Y_M = consensus_model.get_prediction_random_querying(i, threshold)
            result['threshold'] = threshold
            result['data_index'] = t
            results.append(result)
            Y_H_observed.append(Y_O)
            i += 1

            df = pd.DataFrame(results)
            df.to_csv(results_save_dir+"results.csv")



if __name__ == "__main__":
    main()