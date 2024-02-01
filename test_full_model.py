import os

from cmdstanpy import CmdStanModel
import numpy as np

def gen_sample_data(n_items=100):
    # two independent models, two humans correlated w/ m1, one correlated w/ m2
    # generally consensus is humans 1 & 2, i.e. model 1 is more helpful for
    # predicgting consensus
    model_pred = []
    human_pred = []
    for _ in range(n_items):
        underlying_model = np.random.beta(a=0.2, b=0.2)
        underlying_model2 = np.random.beta(a=0.2, b=0.2)
        model_pred.append([
            [underlying_model, 1-underlying_model],
            [underlying_model2, 1-underlying_model2]
        ])
        human_pred.append(
            [np.random.binomial(n=1, p=1-underlying_model)+1,
            np.random.binomial(n=1, p=1-underlying_model)+1,
            np.random.binomial(n=1, p=1-underlying_model2)+1]
        )
    return model_pred, human_pred

def gen_sample_data_k3(n_items=100):
    # two independent models, two humans correlated w/ m1, one correlated w/ m2
    # generally consensus is humans 1 & 2, i.e. model 1 is more helpful for
    # predicgting consensus
    model_pred = []
    human_pred = []
    for _ in range(n_items):
        underlying_model = np.random.beta(a=0.2, b=0.2)
        underlying_model2 = np.random.beta(a=0.2, b=0.2)
        err1 = (1-underlying_model)/2 * 0.0001
        err2 = (1-underlying_model2)/2 * 0.0001
        pvals_1 = [underlying_model, (1-underlying_model)/2 + err1, (1-underlying_model)/2 - err1]
        np.random.shuffle(pvals_1)
        pvals_2 = [underlying_model2, (1-underlying_model2)/2 + err2, (1-underlying_model2)/2 - err2]
        np.random.shuffle(pvals_2)
        model_pred.append([pvals_1, pvals_2])
        human_pred.append(
            [np.argmax(np.random.multinomial(n=1, pvals=pvals_1))+1,
            np.argmax(np.random.multinomial(n=1, pvals=pvals_1))+1,
            np.argmax(np.random.multinomial(n=1, pvals=pvals_2))+1]
        )
    return model_pred, human_pred

def main():
    stan_file = os.path.join(".", "model.stan")
    model = CmdStanModel(stan_file=stan_file)

    n_items = 20
    n_humans = 3
    n_models = 2
    K = 2

    Y_M, Y_H = gen_sample_data(n_items=n_items)

    data = {
        "n_items": n_items,
        "n_models": n_models,
        "n_humans": n_humans,
        "K": K,
       # "eta":1,
        "Y_M": Y_M,
        "Y_H": Y_H
    }

    fit = model.sample(data=data, chains=3, iter_warmup=750, iter_sampling=1000)
    # df = fit.draws_pd()

    computed_quants = model.generate_quantities(data=data, previous_fit=fit)
    df = computed_quants.draws_pd()

    n = (n_models + n_humans) * (K-1)

    print("1----\t\t2----\t\t3----\t\t4----\t\t5----")
    for i in range(1,n+1):
        row = ""
        for j in range(1,n+1):
            corr = np.mean(df['Sigma[{},{}]'.format(i,j)])
            row += str(round(corr,3)) + "\t\t"
        print(row)


    stan_file_inf = os.path.join(".", "expert_inference.stan")
    expert_model = CmdStanModel(stan_file=stan_file_inf)

    data["n_observed_humans"] = 0
    data["unobserved_ind"] = [1,2,3]
    data["n_draws"] = 1000
    data["Y_M_new"] = [[0.1,0.9],[0.9,0.1]]
    data["Y_H_new_real"] = []

    expert_probs = expert_model.generate_quantities(data=data, previous_fit=fit)
    probs_df = expert_probs.draws_pd()

    probs_df.to_csv('probs_df.csv')

    for i in range(n_humans):
        # want the probability to be higher (and about the same) for humans 1 and 2
        print(np.mean(probs_df["p_i_correct[{}]".format(i+1)]))


if __name__ == "__main__":
    main()