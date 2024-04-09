import os
import pandas as pd
import pickle
import time

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
from bisect import bisect
from scipy.stats import entropy

import logging
logger = logging.getLogger('cmdstanpy')
logger.disabled = True

import random
import shutil


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


def get_consensus(arr):
    y = list(arr)
    most = max(list(map(y.count, y)))
    modes = list(set(filter(lambda x: y.count(x) == most, y)))
    if len(modes) > 1:
        return None, modes
    return modes[0], modes

def choose_expert(mvn_fit, consensus_model, exp_dict):

    candidate_experts = exp_dict["unobserved_ind"]
    observed_ind = [i for i in range(1, exp_dict['n_humans']+1) if i not in candidate_experts]
    K = exp_dict["K"]
    hyp_dict = exp_dict.copy()
    hyp_dict['n_observed_humans'] = exp_dict['n_observed_humans'] + 1
    candidate_ees = {}

    for c in candidate_experts:

        print('assessing candidate {}...'.format(c))

        expected_entropy = 0
        hyp_dict['unobserved_ind'] = [i for i in exp_dict['unobserved_ind'] if i != c]

        for j in range(1, K+1):

            # suppose we observe expert i and their vote is j
            Y_O_hyp = exp_dict["Y_O_real"].copy()
            Y_O_hyp.insert(bisect(observed_ind, c), j)
            hyp_dict["Y_O_real"] = Y_O_hyp

            timestamp = str(time.time()).split(".")
            out_dir = "tmp/" + timestamp[1] + str(c) + str(j)

            # get consensus distribution
            hyp_fit = consensus_model.generate_quantities(
                data=hyp_dict, 
                previous_fit=mvn_fit,
                gq_output_dir=out_dir
            )
            consensus_dist_est = np.zeros(K)
            for m in range(K):
                consensus_dist_est[m] = hyp_fit.stan_variable("p_y")[:,m].mean()

            # jth element of the component of L_H corresponding to candidate c
            ind = (c-1)*K + j -1
            p_k = hyp_fit.stan_variable("L_H")[:,ind].mean()

            shutil.rmtree(out_dir)

            e = entropy(consensus_dist_est)
            expected_entropy += p_k * e

        candidate_ees[c] = expected_entropy

    chosen_expert = max(candidate_ees, key=candidate_ees.get)
    insertion_point = bisect(observed_ind, chosen_expert-1)
    return chosen_expert, candidate_ees, insertion_point


def query_next_human(fit, consensus_model, exp_dict, human_labels):
    if len(exp_dict["unobserved_ind"])>1:
        chosen_expert, _, insert_at = choose_expert(fit, consensus_model, exp_dict)
    else:
        chosen_expert = exp_dict["unobserved_ind"][0]
        observed_ind = [i for i in range(1, exp_dict['n_humans']+1) if i != chosen_expert]
        insert_at = bisect(observed_ind, chosen_expert-1)

    # "query" chosen expert
    exp_dict["n_observed_humans"] += 1
    exp_dict["unobserved_ind"] = [i for i in exp_dict["unobserved_ind"] if i!=chosen_expert]
    Y_O_new = exp_dict["Y_O_real"].copy()
    Y_O_new.insert(insert_at, human_labels[chosen_expert-1])
    exp_dict["Y_O_real"] = Y_O_new

    sampled_consensus_dist = consensus_dist(fit, consensus_model, exp_dict)
    pred_y = np.argmax(sampled_consensus_dist) + 1
    return exp_dict, pred_y


def consensus_dist(mvn_fit, consensus_model, exp_dict):
    K = exp_dict["K"]
    fit = consensus_model.generate_quantities(
        data=exp_dict, 
        previous_fit=mvn_fit
    )
    consensus_dist_est = np.zeros(K)
    for m in range(K):
        consensus_dist_est[m] = fit.stan_variable("p_y")[:,m].mean()
    return consensus_dist_est


def main():

    # estimate mu and Sigma
    rerun_model = True

    with open('cifar_10h.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    K = data_dict["K"]

    chains=3
    n_warmup= 500 #1500
    n_sampling= 800 #2000

    n_items = 200 #500
    mini_dict = data_dict.copy()
    mini_dict['Y_M'] = mini_dict['Y_M'][:n_items]
    mini_dict['Y_H'] = mini_dict['Y_H'][:n_items]
    mini_dict['n_items'] = n_items
    mini_dict['eta'] = 1

    fname = "fit_{}_{}_{}".format(n_warmup, n_sampling, n_items)

    n_humans = data_dict['n_humans']
    n = (data_dict['n_models'] + n_humans)*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_mvn_param(mini_dict, fname, chains, n_warmup, n_sampling)
    else:
        fit = from_csv(path='cifar_fits/'+fname, method='sample')

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    consensus_model = CmdStanModel(stan_file=stan_file_inf)

    exp_dict = {
        "n_models" : data_dict['n_models'],
        "n_humans" : data_dict['n_humans'],
        "n_items" : n_items,
        "K" : K,
        "n_observed_humans" : 0,
        "Y_O_real": [],
    }

    n_tests = 100*4
    total = 0; i=-1
    correct = 0; random_correct = 0
    test_results = []; random_results = []

    while total < n_tests:

        human_labels = data_dict["Y_H_new_list"][i+1]
        consensus, _ = get_consensus(human_labels)

        # skip uninteresting examples for now
        if consensus is None:
            i+=1
            continue

        if human_labels[0]==human_labels[1] and human_labels[2]==human_labels[1]:
            i+=1
            continue

        print('running test on row ' + str(i+1))

        # reset exp_dict
        exp_dict["Y_M_new"] = data_dict["Y_M_new_list"][i+1]
        exp_dict['n_observed_humans'] = 0
        exp_dict["Y_O_real"] = []
        exp_dict['unobserved_ind'] = [i for i in range(1, n_humans+1)]

        random_candidate_experts = [i for i in range(n_humans)]
        random_labels = []
        test_result = []
        random_result = []

        model_only_cd = consensus_dist(fit, consensus_model, exp_dict)
        pred_y = np.argmax(model_only_cd) + 1
        naive_pred_y = np.argmax(exp_dict["Y_M_new"][0]) + 1

        if pred_y == consensus:
            test_result.append(1)
            correct += 1
        else:
            test_result.append(0)
        if naive_pred_y == consensus:
            random_result.append(1)
            random_correct += 1
        else:
            random_result.append(0)
        total+=1

        print("chosen:", pred_y)
        print("random:", naive_pred_y)
        print("actual:", consensus)

        for _ in range(n_humans):
            print("querying next human...")
    
            random_expert = np.random.choice(random_candidate_experts)
            random_candidate_experts = [c for c in random_candidate_experts if c != random_expert]
            random_labels.append(human_labels[random_expert])

            exp_dict, pred_y = query_next_human(fit, consensus_model, exp_dict, human_labels)

            _, modes = get_consensus(random_labels)
            naive_pred_y = np.random.choice(modes)

            print("chosen:", pred_y)
            print("random:", naive_pred_y)
            print("actual:", consensus)

            if pred_y == consensus:
                test_result.append(1)
                correct += 1
            else:
                test_result.append(0)
            if naive_pred_y == consensus:
                random_result.append(1)
                random_correct += 1
            else:
                random_result.append(0)

            total += 1

        i+=1
        
        test_results.append(test_result)
        random_results.append(random_result)

        print("random accuracy = ", random_correct/total)
        print("accuracy = ", correct/total)

        random_df = pd.DataFrame(random_results)
        random_df.to_csv("random_results.csv")

        test_df = pd.DataFrame(test_results)
        test_df.to_csv("test_results.csv")

    print("completed testing")

if __name__ == "__main__":
    main()