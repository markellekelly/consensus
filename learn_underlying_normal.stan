
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

    matrix onion(vector R2, vector l, int N) {
        // https://discourse.mc-stan.org/t/nan-results-with-lkj-corr-cholesky/16832
        matrix[N,N] L;
        int start = 1;
        int end = 2;

        L[1,1] = 1.0;
        L[2,1] = 2.0 * R2[1] - 1.0;
        L[2,2] = sqrt(1.0 - square(L[2,1]));
        for(k in 2:(N-1)) {
            int kp1 = k + 1;
            vector[k] l_row = segment(l, start, k);
            real scale = sqrt(R2[k] / dot_self(l_row));
            for(j in 1:k) L[kp1,j] = l_row[j] * scale;
            L[kp1,kp1] = sqrt(1.0 - R2[k]);
            start = end + 1;
            end = start + k - 1;
        }
        return L;
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
    real<lower = 0> eta;

    // model predictions (probabilities)
    array[n_items,n_models,K] real<lower=0,upper=1> Y_M;
    // human predictions (votes)
    array[n_items,n_humans] int<lower=1,upper=K> Y_H;
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

    real<lower = 0> alpha = eta + (N - 2) / 2.0;
    vector<lower = 0>[N-1] shape1;
    vector<lower = 0>[N-1] shape2;
    shape1[1] = alpha;
    shape2[1] = alpha;
    for (n in 2:(N-1)) {
        alpha = alpha - 0.5;
        shape1[n] = n / 2.0;
        shape2[n] = alpha;
    }
}
parameters { 
    vector[choose(N, 2) - 1]  l;
    vector<lower = 0,upper = 1>[N-1] R2;
    row_vector[N] mu;
    real<lower=0> T;
    
    // https://mc-stan.org/docs/stan-users-guide/partially-known-parameters.html
    matrix[n_items,n_humans*(K-1)] Z_H;
}  
transformed parameters {
    matrix[N,N] L_Sigma = onion(R2, l, N);

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
    l ~ normal(0.0, 1.0);
    R2 ~ beta(shape1, shape2);
    mu ~ normal(0.0, 0.1);
    T ~ normal(0, 0.4);
    
    // likelihood of latent scores
    for (i in 1:n_items){
        Z[i] ~ multi_normal_cholesky(mu, L_Sigma);
    }

    // likelihood of (human) votes from latent scores
    for (i in 1:n_items){
        for (j in 1:n_humans){
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
generated quantities {
    // covariance matrix
    matrix[N,N] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);
}