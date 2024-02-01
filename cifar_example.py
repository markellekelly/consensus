import os
import pickle

from cmdstanpy import CmdStanModel, from_csv
import numpy as np


def estimate_sigma(data_dict):
    stan_file = os.path.join(".", "model_onion.stan")
    model = CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=data_dict, 
        chains=3, 
        iter_warmup=800, 
        iter_sampling=1500, 
        show_console=False
    )

    # cifar_fit had chains=3, iter_warmup=750, iter_sampling=1500, sub=30
    # cifar_fit2 chains=3, iter_warmup=800, iter_sampling=1750, sub=40
    # cifar_fit3 3,500,1000,30
    # with open('cifar_fit3.pickle', 'wb') as handle:
    #     pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fit.save_csvfiles('cifar_fits/fit_800_1500_40')

    return fit


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

    sub = 40
    mini_dict = data_dict.copy()
    mini_dict['Y_M'] = mini_dict['Y_M'][:sub]
    mini_dict['Y_H'] = mini_dict['Y_H'][:sub]
    mini_dict['n_items'] = sub
    mini_dict['eta'] = 1

    n = (data_dict['n_models'] + data_dict['n_humans'])*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_sigma(mini_dict)
    else:
        fit = from_csv(path='cifar_fits/fit_800_1500_40', method='sample')

    stan_file_inf = os.path.join(".", "expert_inference_onion.stan")
    expert_model = CmdStanModel(stan_file=stan_file_inf)

    mini_dict["n_observed_humans"] = 0
    mini_dict["unobserved_ind"] = [i for i in range(1, mini_dict["n_humans"] + 1)]
    mini_dict["n_draws"] = 1000
    mini_dict["Y_M_new"] = mini_dict["Y_M_new_list"][0]
    mini_dict["Y_H_new_real"] = []

    Y_H_ground_truth = mini_dict["Y_H_new_list"][0]

    expert_probs = expert_model.generate_quantities(data=mini_dict, previous_fit=fit)

    probs_df = expert_probs.draws_pd()
    probs_df.to_csv('cifar_probs.csv')

    n_humans = mini_dict['n_humans']

    # does choosing the max-prob expert do better than choosing a random expert?
    n_tests = 100
    correct = 0
    random_correct = 0
    total = 0
    for i in range(n_tests):
        print(i)
        mini_dict["Y_M_new"] = mini_dict["Y_M_new_list"][i]

        human_labels = mini_dict["Y_H_new_list"][i]
        consensus = get_consensus(human_labels)

        quants_i = expert_model.generate_quantities(data=mini_dict, previous_fit=fit)
        probs_i = [quants_i.stan_variable("p_i_correct")[:,i].mean() for i in range(n_humans)]
        chosen_expert = np.argmax(probs_i)

        random_expert = np.random.choice(range(n_humans))

        if human_labels[chosen_expert]==consensus:
            correct += 1
        if human_labels[random_expert] == consensus:
            random_correct += 1
        if consensus is not None:
            total += 1
    
    print("random accuracy = ", random_correct/total)
    print("accuracy = ", correct/total)
        



if __name__ == "__main__":
    main()