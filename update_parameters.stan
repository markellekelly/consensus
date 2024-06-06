
functions {
    array[] real additive_logistic(array[] real x){
        // transform from R^{K-1} to the k-simplex
        int len = dims(x)[1];
        array[len+1] real transformed_arr;
        array[len] real exp_array = exp(x);
        real denominator = sum(exp_array)+1;
        // if (denominator == 0){
        //     denominator = 0.000000001;
        // }
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

    vector temperature_scale(vector to_scale, real T, int K) {
        vector[K] scaled;
        real denominator = sum(exp(to_scale / T));
        for (i in 1:K) {
            scaled[i] = exp(to_scale[i]/T) / denominator;
        }
        return scaled;
    }
}
data {
    // metadata
    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> n_items;
    int<lower=1> K; 

    // settings
    int<lower=0, upper=1> use_temp_scaling;
    int<lower=0, upper=1> use_correlations;
    real<lower = 0> eta;

    // model predictions (probabilities)
    array[n_items,n_models,K] real<lower=0,upper=1> Y_M;
    // human predictions (votes), 0 is missing
    array[n_items,n_humans] int<lower=0,upper=K> Y_H;
} 
transformed data {
    int N = (K-1)*(n_models + n_humans);

    array[n_items,n_models*(K-1)] real Z_M_arr;
    for (i in 1:n_items) {
        for (j in 1:n_models) {
            array[K-1] real transformed_arr = inv_additive_logistic(Y_M[i][j]);
            int ind = (j-1)*(K-1);
            for (k in 1:(K-1)) {
                Z_M_arr[i][ind+k] = transformed_arr[k];
            }
        }
    }
    matrix[n_items, n_models*(K-1)] Z_M;
    Z_M = to_matrix(Z_M_arr);
}
parameters { 
    vector<lower=0>[N] L_std;
    cholesky_factor_corr[N] L_Omega;
    row_vector[N] mu;
    real<lower=0> T;
    
    // https://mc-stan.org/docs/stan-users-guide/partially-known-parameters.html
    matrix[n_items,n_humans*(K-1)] Z_H;
}  
transformed parameters {
    matrix[N, N] L_Sigma;
    if (use_correlations==1) {
        L_Sigma = diag_pre_multiply(L_std, L_Omega);
    } else {
        matrix[N, N] identity_mat = diag_matrix(rep_vector(1.0, N));
        L_Sigma = diag_pre_multiply(L_std, identity_mat);
    }

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
    // priors for parameters
    L_std ~ normal(0, 1);
    L_Omega ~ lkj_corr_cholesky(eta);
    mu ~ normal(0.0, 0.1);
    T ~ normal(0, 0.4);
    
    // likelihood of latent scores
    for (i in 1:n_items){
        Z[i] ~ multi_normal_cholesky(mu, L_Sigma);
    }

    // likelihood of (human) votes from latent scores
    for (i in 1:n_items){
        for (j in 1:n_humans){
            // skip missing observations
            if (Y_H[i,j] != 0) {
                vector[K] Pmf;
                array[K-1] real to_transform;
                int ind = (K-1) * (n_models + j - 1);
                for (k in 1:K-1){
                    to_transform[k] = Z[i][ind+k];
                }
                Pmf = to_vector(additive_logistic(to_transform)); 
                if (use_temp_scaling==1) {
                    Y_H[i,j] ~ categorical(temperature_scale(Pmf, T, K));
                } else {
                    Y_H[i,j] ~ categorical(Pmf);
                }
            }
        }
    }
}
generated quantities {
    // covariance matrix
    matrix[N,N] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);
}