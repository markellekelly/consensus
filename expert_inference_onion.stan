functions {

    int real_to_int(real x){
        int current = 0;
        int done = 0;
        int ans = 0;
        while(done != 1) {
            if(x > current && x < current + 2){
                ans = current + 1;
                done = 1;
            } else {
                current += 1;
            }
        }
        return ans;
    }

    int argmax(array[] real x) {
        // assumes values of x are > 0
        real max_val = 0;
        int max_ind = 0;
        int len = dims(x)[1];
        for (i in 1:len) {
            if (x[i] > max_val) {
                max_val = x[i];
                max_ind = i;
            }
        }
        return max_ind;
    }

    array[] real additive_logistic(array[] real x){
        // transform from R^{K-1} to the k-simplex
        int len = dims(x)[1];
        array[len+1] real transformed_arr;
        array[len] real exp_array = exp(x);
        real denominator = sum(exp_array)+1;
        if (denominator == 0){
            denominator = 0.000000001;
        }
        for (i in 1:len){
            transformed_arr[i] = exp_array[i]/denominator;
        }
        transformed_arr[len+1] = 1/denominator;
        return transformed_arr;
    }

    array[] real inv_additive_logistic(array[] real x){
        // transform from the k-simplex to R^{K-1}
        // in binary case: positive numbers correspond to more probability
        // mass on class 0, negative on class 1
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

    matrix reorder_Sigma(int split_point, int N, matrix Sigma){
        // split_point is the last element of the first (current) section
        array[N] int new_order;
        int ind = 1;
        for (i in split_point+1:N) {
            new_order[ind] = i;
            ind += 1;
        }
        for (i in 1:split_point) {
            new_order[ind] = i;
            ind += 1;
        }
        matrix[N,N] new_Sigma;
        for (i in 1:N){
            for (j in 1:N){
                new_Sigma[i][j] = Sigma[new_order[i]][new_order[j]];
            }
        }
        return new_Sigma;
    }
}
data {

    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> n_items;
    int<lower=1> K; 

    real<lower = 0> eta;

    int<lower=0> n_observed_humans;

    // indices of humans that have not been observed (i.e. our candidates)
    array[n_humans - n_observed_humans] int<lower=1, upper=n_humans> unobserved_ind;

    int<lower=1> n_draws;

    // model predictions: probabilities (\in a k-simplex)
    array[n_models,K] real<lower=0,upper=1> Y_M_new;

    // human predictions: votes (\in {1, ... K})
    array[n_observed_humans] real<lower=1,upper=K> Y_H_new_real;

} 
transformed data {

    // convert Y_H_new_real from real to int, because Stan doesn't support length-0 int arrays as data
    array[n_observed_humans] int<lower=1,upper=K> Y_H_new;
    if (n_observed_humans > 0) {
        for (i in 1:n_observed_humans) {
            Y_H_new[i] = real_to_int(Y_H_new_real[i]);
        }
    }

    array[n_models*(K-1)] real Z_M_arr;
    for (j in 1:n_models) {
        array[K-1] real transformed_arr = inv_additive_logistic(Y_M_new[j]);
        int ind = (j-1)*(K-1);
        for (k in 1:(K-1)) {
            Z_M_arr[ind+k] = transformed_arr[k];
        }
    }
    vector[n_models*(K-1)] Z_M;
    Z_M = to_vector(Z_M_arr);

    int N = (K-1)*(n_models + n_humans);

    int len_x2 = (K-1)*n_models; // number observed
    int len_x1 = N - len_x2; // number unobserved
}
parameters { 
    // for compatibility, this block needs to contain the same parameters
    // as those in the main model, although we only use L_Sigma here

    vector[choose(N, 2) - 1]  l;         // do NOT init with 0 for all elements
    vector<lower = 0,upper = 1>[N-1] R2; // first element is not really a R^2 but is on (0,1)  
    
    // see https://mc-stan.org/docs/stan-users-guide/partially-known-parameters.html
    matrix[n_items,n_humans*(K-1)] Z_H;

    matrix[N, N] L_Sigma;

    matrix[n_items, N] Z;
}
generated quantities {

    matrix[N,N] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);

    matrix[N,N] Sigma_r;
    Sigma_r = reorder_Sigma(len_x2, N, Sigma);


    // FORM CONDITIONAL GAUSSIAN
    // see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

    vector[len_x1] mu_1 = rep_vector(0, len_x1);
    vector[len_x2] mu_2 = rep_vector(0, len_x2);

    matrix[len_x1, len_x1] Sigma_11 = block(Sigma_r, 1, 1, len_x1, len_x1);
    matrix[len_x1, len_x2] Sigma_12 = block(Sigma_r, 1, len_x1 + 1, len_x1, len_x2);
    matrix[len_x2, len_x1] Sigma_21 = block(Sigma_r, len_x1 + 1, 1, len_x2, len_x1);
    matrix[len_x2, len_x2] Sigma_22 = block(Sigma_r, len_x1 + 1, len_x1 + 1, len_x2, len_x2);

    matrix[len_x1, len_x2] m_prod = mdivide_right_spd(Sigma_12, Sigma_22);

    vector[len_x1] mu_cond = mu_1 + m_prod * (Z_M - mu_2);

    matrix[len_x1, len_x1] Sigma_cond;
    Sigma_cond = Sigma_11 - m_prod * Sigma_21;


    // ESTIMATE P(Y) VIA SAMPLING

    vector[n_humans*(K-1)] Z_H_new;
    Z_H_new = multi_normal_rng(mu_cond, Sigma_cond);

    array[n_draws] int y_samples;

    for (d in 1:n_draws) {
        array[K] int votes = rep_array(0, K);
        if (n_observed_humans > 0) {
            for (i in 1:n_observed_humans) {
                votes[Y_H_new[i]] += 1;
            }
        }
        for (i in unobserved_ind) {
            // segment of Z_H corresponding to person i
            array[K-1] real Z_H_i;
            Z_H_i = to_array_1d(segment(Z_H_new, (K-1)*(i-1)+1, K-1));
            vector[K] Pmf;
            Pmf = to_vector(additive_logistic(Z_H_i));
            int vote_i = categorical_rng(Pmf);
            votes[vote_i] += 1;
        }
        int y_sample = argmax(votes);
        y_samples[d] = y_sample;
    }

    vector[K] p_y = rep_vector(0, K);
    for (d in 1:n_draws) {
        p_y[y_samples[d]] += 1;
    }
    p_y = p_y ./ n_draws;


    // COMPUTE PROBABILITY OF CORRECT PREDICTION FOR EACH CANDIDATE
    
    array[n_humans - n_observed_humans] real p_i_correct;
    for (i in unobserved_ind) {
        real p_correct = 0;
        array[K-1] real Z_H_i;
        Z_H_i = to_array_1d(segment(Z_H_new, (K-1)*(i-1)+1, K-1));
        array[K] real Z_H_i_trans;
        Z_H_i_trans = additive_logistic(Z_H_i);
        for (k in 1:K) {
            p_correct += Z_H_i_trans[k] * p_y[k];
        }
        p_i_correct[i] = p_correct;
    }

}