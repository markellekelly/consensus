import os
import pickle
import time
import shutil
import logging

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
import pandas as pd
from scipy.stats import entropy

from utils import get_consensus, print_Sigma
from dataset import Dataset

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


def model_consensus(model, exp_dict, mvn_fit, id_str):
    timestamp = str(time.time()).split(".")
    out_dir = "tmp/" + timestamp[0] + id_str
    fit = model.generate_quantities(
        data=exp_dict, 
        previous_fit=mvn_fit,
        gq_output_dir=out_dir
    )
    return fit, out_dir


def get_expert_choice_probabilities(mvn_fit, consensus_model, exp_dict):
    K = exp_dict["K"]
    candidates = exp_dict["unobserved_ind"]
    current_fit, out_dir = model_consensus(
        consensus_model, 
        exp_dict, 
        mvn_fit, 
        "-ec"
    )
    likelihood = current_fit.stan_variable("p_y_k")
    votes = [
        current_fit.stan_variable("Y_U")[:,e] for e in range(len(candidates))
    ]
    shutil.rmtree(out_dir)

    num_votes = len(votes[0])
    probabilities_un = np.zeros((len(candidates), K))
    c_index = 0
    for c in candidates:
        for j in range(1, K+1):
            sample_count = [1 if int(v)==j else 0 for v in votes[c_index]]
            prob_cj = sum(sample_count * likelihood)/num_votes
            probabilities_un[c_index][j-1] = prob_cj
        c_index+=1

    # normalize to get probabilities
    row_sums = probabilities_un.sum(axis=1, keepdims=True)
    probabilities = probabilities_un / row_sums

    return probabilities


def choose_expert(mvn_fit, consensus_model, dataset):

    exp_dict = dataset.get_test_dict()
    candidates= exp_dict["unobserved_ind"]
    observed_ind = [
        i for i in range(1, exp_dict['n_humans']+1) if i not in candidates
    ]
    K = exp_dict["K"]

    candidate_ees = {}

    choice_probabilities = get_expert_choice_probabilities(
        mvn_fit, 
        consensus_model,
        exp_dict
    )
    c_index = 0

    for c in candidates:

        print('assessing candidate {}...'.format(c))
        expected_entropy = 0

        for j in range(1, K+1):

            print('if they vote {}...'.format(j))
            hyp_dict = dataset.get_hypothetical_query_dict(c, j)
            print(hyp_dict)

            # get consensus distribution
            hyp_fit, out_dir = model_consensus(
                model=consensus_model, 
                exp_dict=hyp_dict, 
                mvn_fit=mvn_fit, 
                id_str=str(c) + str(j)
            )
            consensus_dist_est = np.zeros(K)
            for m in range(K):
                consensus_dist_est[m] = hyp_fit.stan_variable("p_y")[:,m].mean()
            shutil.rmtree(out_dir)
            print([i/sum(consensus_dist_est) for i in consensus_dist_est])

            #probability candidate c chooses class j
            p_k = choice_probabilities[c_index][j-1]
            print("{}% chance".format(p_k))

            e = entropy(consensus_dist_est)
            expected_entropy += p_k * e
        
        c_index +=1
        candidate_ees[c] = expected_entropy

    chosen_expert = min(candidate_ees, key=candidate_ees.get)
    print("candidate ees:" + str(candidate_ees))
    print("chose "+ str(chosen_expert))
    return chosen_expert, candidate_ees


# def query_random_human(fit, consensus_model, exp_dict, human_labels):
#     observed_ind = [i for i in range(1, exp_dict['n_humans']+1) if i not in exp_dict["unobserved_ind"]]
#     if len(exp_dict["unobserved_ind"])>1:
#         chosen_expert = np.random.choice(exp_dict["unobserved_ind"])
#     else:
#         chosen_expert = exp_dict["unobserved_ind"][0]
#     insert_at = bisect(observed_ind, chosen_expert-1)

#     # "query" chosen expert
#     exp_dict["n_observed_humans"] += 1
#     exp_dict["unobserved_ind"] = [i for i in exp_dict["unobserved_ind"] if i!=chosen_expert]
#     Y_O_new = exp_dict["Y_O_real"].copy()
#     Y_O_new.insert(insert_at, human_labels[chosen_expert-1])
#     exp_dict["Y_O_real"] = Y_O_new

#     sampled_consensus_dist = consensus_dist(fit, consensus_model, exp_dict)
#     pred_y = np.argmax(sampled_consensus_dist) + 1
#     return exp_dict, pred_y

def query_next_human(fit, consensus_model, dataset):
    
    exp_dict = dataset.get_test_dict()
    if len(exp_dict["unobserved_ind"])>1:
        chosen_expert, _ = choose_expert(fit, consensus_model, dataset)
    else:
        chosen_expert = exp_dict["unobserved_ind"][0]

    # query chosen expert
    exp_dict = dataset.query_dict_update(chosen_expert)
    print(exp_dict)

    sampled_consensus_dist = consensus_dist(fit, consensus_model, exp_dict)
    print("SAMPLED CONSENSUS DIST:")
    print(sampled_consensus_dist)
    uncertainty = 1 - max(sampled_consensus_dist)
    pred_y = np.argmax(sampled_consensus_dist) + 1

    return pred_y, uncertainty

def consensus_dist(mvn_fit, consensus_model, exp_dict):
    K = exp_dict["K"]
    fit, out_dir = model_consensus(consensus_model, exp_dict, mvn_fit, "-cd")
    consensus_dist_est = np.zeros(K)
    for m in range(K):
        consensus_dist_est[m] = fit.stan_variable("p_y")[:,m].mean()
    shutil.rmtree(out_dir)
    consensus_dist_norm = consensus_dist_est/sum(consensus_dist_est)
    return consensus_dist_norm


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

def main():

    logger = logging.getLogger('cmdstanpy')
    logger.disabled = True

    apply_TS = True
    rerun_model = False

    n_items = 300
    chains = 3
    n_warmup= 1500
    n_sampling= 2000

    with open('data/nih.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    dataset = Dataset(
        n_models = data_dict['n_models'],
        n_humans = data_dict['n_humans'],
        n_classes = data_dict['K'],
        n_initialization_items = n_items,
        model_predictions = data_dict['Y_M'],
        human_predictions = data_dict['Y_H'],
        apply_temperature_scaling = apply_TS
    )

    data_dict = dataset.get_init_data_dict()

    K = data_dict["K"]

    fname = "fit2_{}_{}_{}".format(n_warmup, n_sampling, n_items)
    save_dir = 'nih_fits'
    if apply_TS:
        fname += "-ts"

    n_humans = data_dict['n_humans']
    n = (data_dict['n_models'] + n_humans)*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_mvn_param(data_dict, chains, n_warmup, n_sampling)
        fit.save_csvfiles(save_dir+'/'+fname)
    else:
        fit = from_csv(path=save_dir+'/'+fname, method='sample')

    mu, L_Sigma = get_parameters(fit, n)

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    consensus_model = CmdStanModel(stan_file=stan_file_inf)

    # exp_dict["L_Sigma"] = L_Sigma
    # exp_dict["mu"] = mu

    n_tests = 50
    i=0
    test_results = []; random_results = []

    uncertainty_threshold = 0.2

    while i < n_tests:

        dataset.init_test(i)
        consensus = dataset.get_human_consensus(i)

        print('running test on row ' + str(i))

        random_candidate_experts = [i for i in range(n_humans)]
        random_labels = []
        test_result = []
        random_result = []

        model_only_cd = consensus_dist(fit, consensus_model, dataset.get_test_dict())
        uncertainty = 1 - max(model_only_cd)

        pred_y = np.argmax(model_only_cd) + 1
        naive_pred_y = dataset.get_model_prediction(i)

        test_result.append(1 if pred_y == consensus else 0)
        random_result.append(1 if naive_pred_y == consensus else 0)

        print('human labels: ' + str(dataset.Y_H_new[i]))
        print("chosen:", pred_y)
        print("random/model prediction:", naive_pred_y)
        print("actual:", consensus)

        while uncertainty > uncertainty_threshold:
            print("querying next human...")
            random_expert = np.random.choice(random_candidate_experts)
            random_candidate_experts = [c for c in random_candidate_experts if c != random_expert]
            random_labels.append(dataset.Y_H_new[i][random_expert])

            pred_y, uncertainty = query_next_human(fit, consensus_model, dataset)

            _, modes = get_consensus(random_labels)
            naive_pred_y = np.random.choice(modes)

            print("chosen:", pred_y)
            print("random:", naive_pred_y)
            print("actual:", consensus)

            test_result.append(1 if pred_y == consensus else 0)
            random_result.append(1 if naive_pred_y == consensus else 0)

        i+=1
        
        test_results.append(test_result)
        random_results.append(random_result)

        unc_str = str(uncertainty_threshold)
        random_df = pd.DataFrame(random_results)
        random_df.to_csv("nih_results/random_results" + unc_str + ".csv")

        test_df = pd.DataFrame(test_results)
        test_df.to_csv("nih_results/test_results" + unc_str + ".csv")

    print("completed testing")

if __name__ == "__main__":
    main()