import os
import pickle

from cmdstanpy import CmdStanModel, from_csv
import numpy as np
from bisect import bisect
from scipy.stats import entropy

import logging
logger = logging.getLogger('cmdstanpy')
logger.disabled = True

import random


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

def get_mu_Sigma(fit, n):
    mu_estimate = np.zeros(n)
    Sigma_estimate = np.zeros((n,n))
    for i in range(n):
        mu_estimate[i] = fit.stan_variable("mu")[:,i].mean()
        for j in range(n):
            Sigma_estimate[i][j] = fit.stan_variable("Sigma")[:,i,j].mean()

    return mu_estimate, Sigma_estimate  


def get_consensus(arr):
    y = list(arr)
    most = max(list(map(y.count, y)))
    modes = list(set(filter(lambda x: y.count(x) == most, y)))
    if len(modes) > 1:
        return None 
    return modes[0]

def choose_expert(consensus_model, exp_dict):
    candidate_experts = exp_dict["unobserved_ind"]
    observed_ind = [i for i in range(1, exp_dict['n_humans']+1) if i not in candidate_experts]
    K = exp_dict["K"]
    hyp_dict = exp_dict.copy()
    hyp_dict['n_observed_humans'] = exp_dict['n_observed_humans'] + 1
    candidate_ees = {}
    for c in candidate_experts:
        expected_entropy = 0
        hyp_dict['unobserved_ind'] = [i for i in exp_dict['unobserved_ind'] if i != c]
        for j in range(1, K+1):
            # suppose we observe expert i and their vote is j
            Y_O_hyp = exp_dict["Y_O_real"].copy()
            Y_O_hyp.insert(bisect(observed_ind, c), j)
            hyp_dict["Y_O_real"] = Y_O_hyp
            n_chains=3
            hyp_fit = consensus_model.sample(
                data=hyp_dict, 
                chains=n_chains, 
                chain_ids=[int("42" + str(c) +  str(random.randint(1, 100)) +str(j) + str(chain)) for chain in range(n_chains)],
                iter_warmup=200, 
                iter_sampling=500,
                show_progress=False,
                time_fmt='%Y%m%d%H%M%S-' +str(c) + '-' + str(j)
            )
            # jth element of the component of L_H corresponding to candidate c
            ind = (c-1)*K + j -1
            p_k = hyp_fit.stan_variable("L_H")[:,ind].mean()

            consensus_dist = np.zeros(K)
            for m in range(K):
                consensus_dist[m] = hyp_fit.stan_variable("p_y")[:,m].mean()

            e = entropy(consensus_dist)
            expected_entropy += p_k * e
        candidate_ees[c] = expected_entropy
    chosen_expert = max(candidate_ees, key=candidate_ees.get)
    return chosen_expert, candidate_ees


def consensus_dist(consensus_model, exp_dict):
    K = exp_dict["K"]
    n_chains =3
    fit = consensus_model.sample(
        data=exp_dict, 
        chains=n_chains, 
        chain_ids=[int("81" + str(random.randint(1, 100)) + str(chain)) for chain in range(n_chains)],
        iter_warmup=200, 
        iter_sampling=500,
        show_progress=False,
        time_fmt='%Y%m%d%H%M%S-c'
    )
    consensus_dist = np.zeros(K)
    for m in range(K):
        consensus_dist[m] = fit.stan_variable("p_y")[:,m].mean()
    e = entropy(consensus_dist)
    predicted_y = np.argmax(consensus_dist) + 1
    return predicted_y, e


def main():

    # estimate mu and Sigma
    rerun_model = False

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

    n_humans = data_dict['n_humans']
    n = (data_dict['n_models'] + n_humans)*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_mvn_param(mini_dict, fname, chains, n_warmup, n_sampling)
    else:
        fit = from_csv(path='cifar_fits/'+fname, method='sample')

    mu, Sigma = get_mu_Sigma(fit, n)

    stan_file_inf = os.path.join(".", "simulate_consensus.stan")
    consensus_model = CmdStanModel(stan_file=stan_file_inf)

    exp_dict = {
        "n_models" : data_dict['n_models'],
        "n_humans" : data_dict['n_humans'],
        "K" : K,
        "mu": mu,
        "Sigma" : Sigma,
        "n_observed_humans" : 1,
        "Y_M_new": data_dict["Y_M_new_list"][0],
        "Y_O_real": [],
    }

    #candidate_entropies = choose_expert(consensus_model, exp_dict)
    #print(candidate_entropies)

    print("beginning loop....")

    n_tests = 50
    correct = 0
    random_correct = 0
    total = 0; i=0
    while total < n_tests:
        # TODO: edit this to loop over multiple observed experts
        human_labels = data_dict["Y_H_new_list"][i]
        consensus = get_consensus(human_labels)

        observed_human = np.random.choice(range(n_humans))

        if consensus is None:
            i+=1
            continue

        if human_labels[0]==human_labels[1] and human_labels[2]==human_labels[1]:
            i+=1
            continue

        print(total)
        exp_dict["n_observed_humans"] = 1
        exp_dict["unobserved_ind"] = [i for i in range(1, exp_dict["n_humans"] + 1) if i!= observed_human+1]
        exp_dict["Y_M_new"] = data_dict["Y_M_new_list"][i]
        exp_dict["Y_O_real"] = [human_labels[observed_human]]
 
        chosen_expert, _ = choose_expert(consensus_model, exp_dict)
        random_expert = np.random.choice(range(n_humans))

        # TODO: choose second expert

        # print('chosen:', chosen_expert)
        # print('random:', random_expert)

        pred_y, _ = consensus_dist(consensus_model, exp_dict)

        print("chosen:", pred_y)
        print("random:", human_labels[random_expert])
        print("actual:", consensus)

        if pred_y == consensus:
            correct += 1
        if human_labels[random_expert] == consensus:
            random_correct += 1
        total += 1; i+=1

        if total % 10 ==0:
            print("random accuracy = ", random_correct/total)
            print("accuracy = ", correct/total)
    
    print("random accuracy = ", random_correct/total)
    print("accuracy = ", correct/total)


if __name__ == "__main__":
    main()