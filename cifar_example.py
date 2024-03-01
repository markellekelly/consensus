import os
import pickle

from cmdstanpy import CmdStanModel, from_csv
import numpy as np


def estimate_sigma(data_dict, fname=None, chains=3, iter_warmup=500, iter_sampling=1000):
    stan_file = os.path.join(".", "model_onion.stan")
    model = CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=data_dict, 
        chains=chains, 
        iter_warmup=iter_warmup, 
        iter_sampling=iter_sampling, 
        show_console=False
    )

    # cifar_fit had chains=3, iter_warmup=750, iter_sampling=1500, sub=30
    # cifar_fit2 chains=3, iter_warmup=800, iter_sampling=1750, sub=40
    # cifar_fit3 3,500,1000,30
    # with open('cifar_fit3.pickle', 'wb') as handle:
    #     pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if fname:
        fit.save_csvfiles('cifar_fits/'+fname)

    return fit


def get_consensus(arr):
    y = list(arr)
    most = max(list(map(y.count, y)))
    modes = list(set(filter(lambda x: y.count(x) == most, y)))
    if len(modes) > 1:
        return None 
    return modes[0]

def main():
    rerun_model = False

    with open('cifar_10h.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    n_items = 500
    mini_dict = data_dict.copy()
    mini_dict['Y_M'] = mini_dict['Y_M'][:n_items]
    mini_dict['Y_H'] = mini_dict['Y_H'][:n_items]
    mini_dict['n_items'] = n_items
    mini_dict['eta'] = 1

    K = data_dict["K"]

    chains=5
    iter_warmup=1500
    iter_sampling=2000

    fname = "fit_{}_{}_{}".format(iter_warmup, iter_sampling, n_items)

    n = (data_dict['n_models'] + data_dict['n_humans'])*(data_dict['K'] - 1)

    if rerun_model:
        fit = estimate_sigma(mini_dict, fname, chains, iter_warmup, iter_sampling)
    else:
        fit = from_csv(path='cifar_fits/'+fname, method='sample')

    Sigma_estimate = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Sigma_estimate[i][j] = fit.stan_variable("Sigma")[:,i,j].mean()

    stan_file_inf = os.path.join(".", "simulate_y.stan")
    expert_model = CmdStanModel(stan_file=stan_file_inf)

    exp_dict = {
        "n_models" : data_dict['n_models'],
        "n_humans" : data_dict['n_humans'],
        "K" : K,
        "Sigma" : Sigma_estimate,
        "n_observed_humans" : 1,
        "unobserved_ind" : [i for i in range(1, mini_dict["n_humans"] + 1)],
        "n_draws" : 500,
        "simulate_y" : 1
    }

    # expert_probs = expert_model.generate_quantities(data=mini_dict, previous_fit=fit)

    # probs_df = expert_probs.draws_pd()
    # probs_df.to_csv('cifar_probs.csv')

    n_humans = data_dict['n_humans']

    stan_file_i = os.path.join(".", "simulate_y.stan")
    stan_file_z = os.path.join(".", "estimate_z.stan")

    # does choosing the max-prob expert do better than choosing a random expert?
    n_tests = 250
    correct = 0
    random_correct = 0
    total = 0; i=0
    while total < n_tests:

        print(total)
        exp_dict["Y_M_new"] = data_dict["Y_M_new_list"][i]

        human_labels = data_dict["Y_H_new_list"][i]
        consensus = get_consensus(human_labels)

        if consensus is None:
            i+=1
            continue

        if human_labels[0]==human_labels[1] and human_labels[2]==human_labels[1]:
            i+=1
            continue

        # estimate z_U^H (todo: use simulate_y if n_observed > 0)
        model_z = CmdStanModel(stan_file=stan_file_z)

        z_dict = {
            "n_models": data_dict['n_models'],
            "n_humans": data_dict['n_humans'],
            "K": K,
            "Sigma": Sigma_estimate,
            "Y_M_new": data_dict["Y_M_new_list"][i]
        }

        fit_z = model_z.sample(
            data=z_dict, 
            chains=3, 
            iter_warmup=500, 
            iter_sampling=750, 
            show_console=False
        )

        latent_probs = np.zeros((n_humans,K))
        for h in range(n_humans):
            for j in range(K):
                latent_probs[h][j] = fit_z.stan_variable("latent_probs")[:,h,j].mean()


        #observed_expert = np.random.choice(range(n_humans))
        exp_dict["Y_M_new"] = data_dict["Y_M_new_list"][i]

        expected_entropy = [0]*n_humans

        for expert_ind in range(n_humans):
            for expert_pred in range(K):
                model_i = CmdStanModel(stan_file=stan_file_i)

                exp_dict["Y_H_new_real"] = [expert_pred+1]
                exp_dict["n_observed_humans"] = 1
                exp_dict["unobserved_ind"] = [i+1 for i in range(n_humans) if i != expert_ind]

                fit_i = model_i.sample(
                    data=exp_dict, 
                    chains=3, 
                    iter_warmup=500, 
                    iter_sampling=750, 
                    show_console=False
                )

                neg_entropy = 0
                for k in range(K):
                    p_y_k = fit_i.stan_variable("p_y")[:,k].mean()
                    neg_entropy += p_y_k * np.log(p_y_k)
                
                expected_entropy[expert_ind] += -1*neg_entropy * latent_probs[expert_ind][k]

        # print(human_labels)

        chosen_expert = np.argmin(expected_entropy)

        random_expert = np.random.choice(range(n_humans))

        # print('chosen:', chosen_expert)
        # print('random:', random_expert)

        if human_labels[chosen_expert] == consensus:
            correct += 1
        if human_labels[random_expert] == consensus:
            random_correct += 1
        total += 1; i+=1
    
    print("random accuracy = ", random_correct/total)
    print("accuracy = ", correct/total)
        


if __name__ == "__main__":
    main()