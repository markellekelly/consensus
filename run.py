import os
import pickle
import logging
import shutil
import sys

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
import pandas as pd

from utils import get_consensus, print_Sigma
from dataset import TestDataset
from consensus_model import ConsensusModel

# parameters for experiment/stan
N_TESTS = 250
N_CHAINS = 3
N_WARMUP = 1500
N_SAMPLES = 2000
STAN_CONSENSUS_FNAME = "simulate_consensus.stan"
STAN_UPDATE_FNAME = "update_parameters.stan"


def get_fnames(
        dataset_name, 
        start_point, 
        use_ts, 
        use_corr, 
        dist_shift, 
        shuffle_num,
        add=""
    ):

    model_id = dataset_name + '_start' + str(start_point) + "_shuf" + str(shuffle_num) + "_"
    ts_str = "" if use_ts else "_nots"
    corr_str = "" if use_corr else "_nocorr"
    ds_str = "_ds" if dist_shift else ""
    results_save_dir = "results/" + dataset_name + "/"
    results_save_f = "start" + str(start_point)
    results_save_f += ts_str + corr_str + ds_str
    if shuffle_num != "":
        results_save_f += "_shuf"+ shuffle_num
    results_save_f += ts_str + corr_str + ds_str + "_"
    results_path = results_save_dir + results_save_f + add
    data_file = 'data/{}/data{}{}.pickle'.format( 
        dataset_name, ds_str, shuffle_num
    )

    return data_file, results_path, model_id

def update_param(data_dict, id_str):
    model = CmdStanModel(stan_file=os.path.join(".", STAN_UPDATE_FNAME))
    out_dir = "tmp/" + "mvn_param" + id_str
    fit = model.sample(
        data=data_dict, 
        chains=N_CHAINS, 
        output_dir=out_dir,
        iter_warmup=N_WARMUP, 
        iter_sampling=N_SAMPLES, 
        show_console=False
    )
    return fit, out_dir

def to_update(t, dist_shift = False):
    if dist_shift:
        if t < 20 or (t > 125 and t < 145) or (t % 10 == 0):
            return True
        return False
    if t < 20 or (t % 10 == 0 and t < 100) or (t % 50 == 0 and t < 250):
        return True
    return False


def run_experiment_with_threshold(
        data_dict, 
        threshold,
        start_point, 
        use_temp_scaling, 
        use_correlations,
        distribution_shift,
        sliding_window,
        eta,
        id_,
        results_path
    ):
    results = []
    fit_dir = None
    dataset = TestDataset(
        n_models = data_dict['n_models'],
        n_humans = data_dict['n_humans'],
        n_classes = data_dict['K'],
        model_predictions = [data_dict['Y_M'][0]],
        human_predictions = [data_dict['Y_H'][0]],
        model_predictions_test = data_dict['Y_M'][start_point:],
        human_predictions_test = data_dict['Y_H'][start_point:],
        use_temp_scaling = use_temp_scaling,
        use_correlations = use_correlations,
        eta = eta
    )

    for t in range(N_TESTS):
        # skip some updates for efficiency
        if to_update(t, distribution_shift):
            if fit_dir is not None:
                shutil.rmtree(fit_dir)
            if sliding_window:
                dataset.truncate(sliding_window)
            init_dict = dataset.get_init_stan_dict()
            fit_new, fit_dir = update_param(
                    init_dict,
                    id_str = str(t) + "_" + id_
            )
            stan_model = CmdStanModel(
                stan_file=os.path.join(".", STAN_CONSENSUS_FNAME)
            )
            consensus_model = ConsensusModel(
                dataset, 
                fit_new, 
                stan_model, 
                id_+str(t)
            )
        
        result = consensus_model.get_prediction(0, threshold)
        result['threshold'] = threshold
        result['data_index'] = t + start_point
        results.append(result)

        pd.DataFrame(results).to_csv(results_path + str(threshold) + ".csv")

    if fit_dir is not None:
        shutil.rmtree(fit_dir)

def main(dataset_name, start_point, suffix, shuffle_num):

    eta = 0.75
    use_temp_scaling = 1 # 0 or 1, default 1
    use_correlations = 1 # 0 or 1, default 1
    distribution_shift = 0 # 0 or 1, default 0
    sliding_window = None # None or int < 250 (50 used for experiments), default None

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    data_file, results_path, id_ = get_fnames(
        dataset_name, 
        start_point,
        use_temp_scaling,
        use_correlations,
        distribution_shift,
        shuffle_num,
        add=suffix
    )

    with open(data_file, 'rb') as handle:
        data_dict = pickle.load(handle)

    thresholds = list(np.arange(0, 0.01, 0.005)) + \
        list(np.arange(0.01, 0.05, 0.01)) + \
        list(np.arange(0.025, 0.05, 0.3)) + \
        list(np.arange(0.05, 0.3, 0.6))

    for threshold in thresholds:
        attempts = 0
        max_retries = 5

        while attempts < max_retries:
            try:
                run_experiment_with_threshold(
                    data_dict, 
                    threshold,
                    start_point, 
                    use_temp_scaling, 
                    use_correlations,
                    distribution_shift,
                    sliding_window,
                    eta,
                    id_,
                    results_path
                )
                attempts = max_retries
            except Exception as e:
                attempts += 1
                print(f"Failed {attempts}/{max_retries} for t={threshold}: {e}")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 run.py <dataset_name> <start_point> <suffix> <shuffle_num>")
        sys.exit(1)

    dataset_name = sys.argv[1] # options:  "nih", "cifar", "imagenet", "chaoyang"
    start_point = int(sys.argv[2]) # 0, 250, 500, ...
    suffix = sys.argv[3] # additional thresholds: "-at","-at2" eta: "-eta01"
    shuffle_num = sys.argv[4] # "", "1",  "2", or "3"

    main(dataset_name, start_point, suffix, shuffle_num)