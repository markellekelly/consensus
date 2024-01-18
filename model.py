import stan
import pandas as pd
import numpy as np
import arviz as az
import xarray
import sys

stan_code = """
functions {

    array[] real additive_logistic(array[] real x){
        // transform from R^{K-1} to the k-simplex
        int len = dims(x)[1];
        array[len+1] real transformed_arr;
        array[len] real exp_array = exp(x);
        real denominator = sum(exp_array)+1;
        for (i in 1:len){
            transformed_arr[i] = exp_array[i]/denominator;
        }
        transformed_arr[len+1] = 1/denominator;
        return transformed_arr;
    }

    array[] real inv_additive_logistic(array[] real x){
        // transform from the k-simplex to R^{K-1}
        int new_len = dims(x)[1]-1;
        array[new_len] real transformed_arr;
        real last_element = x[new_len+1];
        for (i in 1:new_len){
            real frac = x[i]/last_element;
            // if x[i] is 0, use a very small number instead to avoid log 0
            transformed_arr[i] = log(max([frac, 0.000000001]));
        }
        return transformed_arr;
    }

}
data {

    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> n_items;
    int<lower=1> K; 

    // model predictions: probabilities (\in a k-simplex)
    array[n_items,n_models,K] real<lower=0,upper=1> Y_M;

    // human predictions: votes (\in {1, ... K})
    array[n_items,n_humans] int<lower=1,upper=K> Y_H;

} 
transformed data {

    array[n_items,n_models*(K-1)] real Z_M_arr;
    for (i in 1:n_items) {
        for (j in 1:n_models) {
            array[K-1] real transformed_arr = inv_additive_logistic(Y_M[i][j]);
            for (k in 1:(K-1)) {
                int ind = (j-1)*(K-1)+k;
                Z_M_arr[i][ind] = transformed_arr[k];
            }
        }
    }
    matrix[n_items, n_models*(K-1)] Z_M;
    Z_M = to_matrix(Z_M_arr);

    int N = (K-1)*(n_models + n_humans);

}
parameters { 

    vector<lower=0>[N] L_std;
    cholesky_factor_corr[N] L_Omega;

    // see https://mc-stan.org/docs/stan-users-guide/partially-known-parameters.html
    matrix[n_items,n_humans*(K-1)] Z_H;

}  
transformed parameters {

    matrix[N, N] L_Sigma = diag_pre_multiply(L_std, L_Omega);
    
    matrix[n_items, N] Z;
    for (i in 1:n_items){
        for (j in 1:n_models*(K-1)){
            Z[i][j] = Z_M[i][j];
        }
        for (k in 1:n_humans*(K-1)){
            Z[i][k+(n_models*(K-1))] = Z_H[i][k];
        }
    }

}
model {

    L_Omega ~ lkj_corr_cholesky(1);
    L_std ~ normal(0, 0.5);
    
    // fix mu at 0
    row_vector[N] mu;
    mu = rep_row_vector(0, N);

    for (i in 1:n_items){
        Z[i] ~ multi_normal_cholesky(mu, L_Sigma);
    }

    // get human one-hot predictions from latent categorical dist
    for (i in 1:n_items){
        for (j in 1:n_humans){
            vector[K] Pmf;
            array[K-1] real to_transform;
            int ind = (K-1)*j;
            for (k in 1:K-1){
                to_transform[k] = Z[i][ind+k];
            }
            Pmf = to_vector(additive_logistic(to_transform)); 
            Y_H[i,j] ~ categorical(Pmf);
        }
    }

}
generated quantities {

    matrix[N,N] Omega;
    Omega = multiply_lower_tri_self_transpose(L_Omega);

}
"""

def gen_sample_data(n_items=100, high_corr=True):
    model_pred = []
    human_pred = []
    for _ in range(n_items):
        underlying_model = np.random.beta(a=0.2, b=0.2)
        if high_corr:
            underlying_human = underlying_model
        else:
            underlying_human = np.random.random()
        model_pred.append([[underlying_model, 1-underlying_model]])
        human_pred.append([np.random.binomial(n=1, p=1-underlying_human)+1])
    return model_pred, human_pred


def run_model(code, data, num_chains=3, num_warmup=800, num_samples=1500):
    '''build and fit model given data dict'''

    posterior = stan.build(code, data=data)
    
    fit = posterior.sample(num_chains=num_chains,
                            num_warmup=num_warmup, 
                            num_samples=num_samples)

    return fit


def full_model(data):
    '''
    learn a multidimensional IRT model, trained and evaluated on dat
    return the fit dataframe, waic, and loo scores
    '''
    global stan_code

    fit = run_model(stan_code, data, num_chains=3, num_warmup=1000, num_samples=1000)
    #stan_data = gen_arviz_data(data, fit, participants)

    out = fit.to_frame()
    #waic = az.waic(stan_data)
    #loo = az.loo(stan_data)

    return out, None, None

if __name__ == "__main__":

    fname= "test"

    Y_M, Y_H = gen_sample_data(n_items=250, high_corr=False)

    data = {
        "n_items": 250,
        "n_models": 1,
        "n_humans": 1,
        "K": 2,
        "Y_M": Y_M,
        "Y_H": Y_H
    }

    out, waic, loo = full_model(data)

    print("corr = ", np.mean(out['Omega.2.1']))
        
    # save results
    out.to_csv(fname + ".csv")
    # with open(fname+"_metrics.csv", "w") as f:
    #     f.write(str(waic))
    #     f.write("\n")
    #     f.write(str(loo))
