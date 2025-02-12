import os
import logging
import shutil
from cmdstanpy import CmdStanModel, from_csv
import pandas as pd
from dataset import TestDataset
from consensus_model import ConsensusModel

STAN_CONSENSUS_FNAME = "simulate_consensus.stan"
STAN_UPDATE_FNAME = "update_parameters.stan"
N_CHAINS = 3
N_WARMUP = 1500
N_SAMPLES = 2000

logger = logging.getLogger('cmdstanpy')
logger.disabled = True


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
        # increase update frequency when shift occurs at t=125
        if t < 20 or (t > 125 and t < 145) or (t % 10 == 0):
            return True
        return False
    if t < 20 or (t % 10 == 0 and t < 100) or (t % 50 == 0 and t < 250):
        return True
    return False


def run_experiment_with_threshold(
        data_dict, 
        threshold,
        model_id,
        results_path,
        start_point=0, 
        n_tests=250,
        eta=0.75,
        use_temp_scaling=1, 
        use_correlations=1,
        distribution_shift=0,
        sliding_window=None
    ):
    '''
    For `n_tests` iterations, run a consensus experiment, using our model to
    choose which and how many experts to query, and updating its parameters
    periodically.

    Required arguments:
    `data_dict`: a dictionary containing keys `n_models`, `n_humans`, and `K`, 
        with integer values for each (where K is the number of classes), as
        well as lists `Y_M` (model probability estimates, shape (n_examples,
        n_models, K)) and `Y_H` (expert votes, ints 1,...K, shape (n_examples,
        n_experts))
    `threshold`: a float between 0 and 1 specifying the maximum uncertainty
        allowable. The model will continue querying experts until its
        uncertainty falls below this value.
    `model_id`: a unique identifying string for the model used to avoid
        conflicts in Stan temporary directories.
    `results_path`: a path to a file name at which to save a .csv file of results

    '''
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

    for t in range(n_tests):
        # skip some updates for efficiency
        if to_update(t, distribution_shift):
            if fit_dir is not None:
                shutil.rmtree(fit_dir)
            if sliding_window:
                dataset.truncate(sliding_window)
            init_dict = dataset.get_init_stan_dict()
            fit_new, fit_dir = update_param(
                    init_dict,
                    id_str = str(t) + "_" + model_id
            )
            stan_model = CmdStanModel(
                stan_file=os.path.join(".", STAN_CONSENSUS_FNAME)
            )
            consensus_model = ConsensusModel(
                dataset, 
                fit_new, 
                stan_model, 
                model_id+str(t)
            )
        
        result = consensus_model.get_prediction(0, threshold)
        result['threshold'] = threshold
        result['data_index'] = t + start_point
        results.append(result)

        pd.DataFrame(results).to_csv(results_path)

    if fit_dir is not None:
        shutil.rmtree(fit_dir)

