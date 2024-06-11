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


def get_fnames(dataset_name, noisy, start_point, use_ts, use_corr):

    model_id = dataset_name + str(noisy) + '_fit_' + str(start_point)
    nois_str = "" if noisy == 0 else "_noisy"
    if noisy == 2:
        nois_str += "2"
    ts_str = "" if use_ts else "_nots"
    corr_str = "" if use_corr else "_nocorr"
    results_save_dir = "results/" + dataset_name + "/"
    results_save_f = "start" + str(start_point) + nois_str + ts_str + corr_str
    results_path = results_save_dir + results_save_f + ".csv"
    data_file = 'data/{}/data{}.pickle'.format(dataset_name, nois_str)

    return data_file, results_path, model_id

def update_param(data_dict, chains, n_warmup, n_sampling, id_str):
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
    dataset_name = "imagenet"
    # noisy options: 0, 1 (for cifar also 2)
    noisy = 0
    n_tests = 250
    start_point = 500

    chains = 3
    n_warmup= 1500
    n_sampling= 2000
    
    eta = 0.75
    use_temp_scaling = 1
    use_correlations = 1

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    data_file, results_path, id_ = get_fnames(
        dataset_name, 
        noisy, 
        start_point,
        use_temp_scaling,
        use_correlations
    )

    with open(data_file, 'rb') as handle:
        data_dict = pickle.load(handle)

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    results = []

    thresholds = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    thresholds = [0.4, 0.2, 0.1, 0.05, 0]
    for threshold in thresholds:

        dataset = TestDataset(
            n_models = data_dict['n_models'],
            n_humans = data_dict['n_humans'],
            n_classes = data_dict['K'],
            model_predictions = [data_dict['Y_M'][0]],
            human_predictions = [data_dict['Y_H'][0]],
            model_predictions_test = data_dict['Y_M'][start_point:],
            human_predictions_test = data_dict['Y_H'][start_point:],
            use_temp_scaling = use_temp_scaling,
            eta = eta
        )

        for t in range(n_tests):
            # update after every data point for the first 20 iterations, 
            # then every 10 until 100, then every 50
            if t < 20 or (t % 10 == 0 and t < 100) or (t % 50 == 0 and t < 250):
                init_dict = dataset.get_init_stan_dict()
                fit_new, _ = update_param(
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

            result = consensus_model.get_prediction(0, threshold)
            # result = consensus_model.get_prediction_random_querying(0, threshold)
            result['threshold'] = threshold
            result['data_index'] = t + start_point
            results.append(result)

            pd.DataFrame(results).to_csv(results_path)


if __name__ == "__main__":
    main()