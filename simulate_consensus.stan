functions {
    int i_in(int search_val, array[] int search_arr) {
        // return 1 if search_val is in search_arr else 0
        for (p in 1:(size(search_arr))) {
            if (search_arr[p]==search_val) {
                return 1;
            } 
        }
        return 0;
    }

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

    int argmax_with_ties_rng(array[] real x) {
        // get argmax, break ties randomly (assumes values > 0)
        real max_val = 0;
        int num_maxs = 1;
        int len = dims(x)[1];
        for (i in 1:len) {
            if (x[i] > max_val) {
                max_val = x[i];
                num_maxs = 1;
            } else if (x[i] == max_val) {
                num_maxs += 1;
            }
        }
        array[num_maxs] int max_inds;
        int j = 1;
        for (i in 1:len) {
            if (x[i] == max_val) {
                max_inds[j] = i;
                j += 1;
            }
        }
        if (num_maxs == 1){
            return max_inds[1];
        }
        // break ties
        vector[num_maxs] pmf = rep_vector(1./num_maxs, num_maxs);
        int winning_element = categorical_rng(pmf);
        return max_inds[winning_element];
    }

    int mode_rng(array[] int x, int n) {
        // get mode, break ties randomly (assumes values are ints 1, ..., n)
        array[n] real counts = rep_array(0, n);
        int len = dims(x)[1];
        for (i in 1:len){
            counts[x[i]] += 1.0;
        }
        return argmax_with_ties_rng(counts);
    }

    array[] real additive_logistic(array[] real x){
        // transform from R^{K-1} to the k-simplex
        int len = dims(x)[1];
        array[len+1] real transformed_arr;
        // handle the case where elements of x are too large to exponentiate
        array[len] real trunc_x;
        for (i in 1:len){
            trunc_x[i] = min([x[i], 10]);
        }
        array[len] real exp_array = exp(trunc_x);
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

    vector temperature_scale(vector to_scale, real T, int K) {
        vector[K] scaled;
        real denominator = sum(exp(to_scale / T));
        for (i in 1:K) {
            scaled[i] = exp(to_scale[i]/T) / denominator;
        }
        return scaled;
    }

    array[] int expand_to_k(array[] int to_expand, int K) {
        // convert an array of agent indicies to agent + class-wise indicies
        // e.g., for K = 3, N = 3: [1,3] -> [1,2,5,6] [2,1] -> [3,4,1,2]
        int n_agents = dims(to_expand)[1];
        array[n_agents*(K-1)] int expanded;
        int ind = 1;
        for (i in to_expand) {
            for (k in 1:(K-1)) {
                expanded[ind] = (i-1)*(K-1) + k;
                ind += 1;
            }
        }
        return expanded;
    }

    matrix reorder_Sigma(matrix Sigma, array[] int new_order){
        // reorders Sigma to follow new_order
        int N = dims(new_order)[1];
        matrix[N,N] new_Sigma;
        for (i in 1:N){
            for (j in 1:N){
                new_Sigma[i][j] = Sigma[new_order[i]][new_order[j]];
            }
        }
        return new_Sigma;
    }

    vector reorder_row_vector(row_vector to_reorder, array[] int new_order){
        // reorders row vector to a vector following new_order
        int N = dims(new_order)[1];
        vector[N] new_vector;
        for (i in 1:N){
            new_vector[i] = to_reorder[new_order[i]];
        }
        return new_vector;
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
    real<lower=0> eta;

    // additional metadata (not used in update_parameters)
    int<lower=0> n_observed_humans;

    // indices of humans that have not been observed (candidates for querying)
    array[n_humans - n_observed_humans] int<lower=1, upper=n_humans> unobserved_ind;

    // model predictions (probabilities)
    array[n_models,K] real<lower=0,upper=1> Y_M_new;
    // human predictions (votes) (as reals, 0-length int arrays not supported)
    array[n_observed_humans] real<lower=1,upper=K> Y_O_real;
} 
transformed data {
    // convert Y_O_real to ints
    array[n_observed_humans] int<lower=1,upper=K> Y_O;
    if (n_observed_humans > 0) {
        for (i in 1:n_observed_humans) {
            Y_O[i] = real_to_int(Y_O_real[i]);
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
    int n_unobserved_humans = n_humans - n_observed_humans;

    // arrays of indices for convenience
    array[n_models] int model_ind;
    array[n_humans] int human_ind;
    array[n_observed_humans] int observed_true_ind;
    array[n_unobserved_humans] int unobserved_true_ind;

    for (i in 1:n_models) {
        model_ind[i] = i;
    }
    int u_ind = 1;
    int o_ind = 1;
    for (i in 1:n_humans) {
        human_ind[i] = n_models + i;
        if (i_in(i, unobserved_ind)) {
            unobserved_true_ind[u_ind] = n_models + i;
            u_ind += 1;
        } else {
            observed_true_ind[o_ind] = n_models + i;
            o_ind += 1;
        }
    }

    // sizes for the conditional Gaussian | Z_M
    int s1 = n_humans*(K-1); // first block: unobserved (to estimate)
    int s2 = n_models*(K-1); // second block: observed (to condition on)
} 
parameters {
    // parameters from learn_underlying_normal (for compatibility)
    vector<lower=0>[N] L_std;
    matrix[N,N] L_Omega;
    row_vector[N] mu;
    real<lower=0> T;

    matrix[n_items,n_humans*(K-1)] Z_H;

    matrix[N,N] L_Sigma;
    matrix[n_items, N] Z;
}
generated quantities {
    // covariance matrix
    matrix[N,N] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);

    // reorder mu and Sigma so unobserved (human) predictions come first
    array[n_models + n_humans] int new_order = append_array(human_ind, model_ind); 
    array[N] int new_order_ind = expand_to_k(new_order, K);
    vector[N] mu_reordered = reorder_row_vector(mu, new_order_ind);
    matrix[N,N] Sigma_reordered = reorder_Sigma(Sigma, new_order_ind);

    // get mu' and Sigma' for the conditional Gaussian | Z_M
    vector[s1] mu_1 = segment(mu_reordered, 1, s1);
    vector[s2] mu_2 = segment(mu_reordered, s1 + 1, s2);

    matrix[s1, s1] Sigma_11 = block(Sigma_reordered, 1, 1, s1, s1);
    matrix[s1, s2] Sigma_12 = block(Sigma_reordered, 1, s1 + 1, s1, s2);
    matrix[s2, s1] Sigma_21 = block(Sigma_reordered, s1 + 1, 1, s2, s1);
    matrix[s2, s2] Sigma_22 = block(Sigma_reordered, s1 + 1, s1 + 1, s2, s2);

    matrix[s1, s2] m_prod = mdivide_right_spd(Sigma_12, Sigma_22);

    vector[s1] mu_cond = mu_1 + m_prod * (Z_M - mu_2);
    matrix[s1, s1] Sigma_cond;
    Sigma_cond = Sigma_11 - m_prod * Sigma_21;

    // draw Z_H from the conditional Gaussian | Z_M
    vector[s1] Z_H_draw = multi_normal_rng(mu_cond, Sigma_cond);

    // draw sampled votes for the unobserved humans Y_U | Z_U
    array[n_unobserved_humans] int<lower=1,upper=K> Y_U;
    int j = 1;
    vector[K] Pmf_draw;
    for (i in 1:n_humans) {
        array[K-1] real Z_i_draw = to_array_1d(segment(Z_H_draw, ((K-1)*(i-1))+1, K-1));
        // to do: handle the case where Z_i is too big for the exponential in additive_logistic
        Pmf_draw = to_vector(additive_logistic(Z_i_draw));  
        if (use_temp_scaling==1) {
            Pmf_draw = temperature_scale(Pmf_draw, T, K);
        }
        if (i_in(i, unobserved_ind)) {
            Y_U[j] = categorical_rng(Pmf_draw);
            j += 1;
        }
    }
    
    // get y_mode and likelihood of Y_O | Z_O
    int<lower=1,upper=K> y_mode;
    real p_y_k = 1;
    if (n_observed_humans >= 1) {
        // get mode of sampled Y_U and actual Y_O
        array[n_humans] int Y_H = append_array(Y_O, Y_U);
        y_mode = mode_rng(Y_H, K);

        // weight p_y_k by likelihood of Y_O | Z_O
        int observed_count = 1;
        for (i in 1:n_humans) {
    
            if (i_in(i+n_models, observed_true_ind)) {
                // vote of expert i
                int Y_i = Y_O[observed_count];
                // sampled confidence of expert i
                array[K-1] real Z_i = to_array_1d(segment(Z_H_draw, (K-1)*(i-1)+1, K-1));
                vector[K] Pmf = to_vector(additive_logistic(Z_i)); 
                if (use_temp_scaling==1) {
                    p_y_k *= temperature_scale(Pmf, T, K)[Y_i];
                } else {
                    p_y_k *= Pmf[Y_i];
                }
                observed_count +=1;
            }
            
        }
    } else {
        // get mode of Y_U
        y_mode = mode_rng(Y_U, K);
    }

    // assign p_y_k to the corresponding element of p_y
    vector[K] p_y = rep_vector(0, K);
    p_y[y_mode] = p_y_k;
}