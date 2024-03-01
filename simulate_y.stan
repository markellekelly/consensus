functions {

    int i_in(int pos, array[] int pos_var) {
        // if(i_in(3,{1,2,3,4})) will evaluate as 1
        for (p in 1:(size(pos_var))) {
            if (pos_var[p]==pos) {
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

    array[] int expand_to_k(array[] int to_expand, int K) {
        // convert an array of agent indicies to indicies over the full N
        // e.g. for K = 3, N = 3: [1,3] -> [1,2,5,6] [2,1] -> [3,4,1,2]
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

    matrix reorder_Sigma(matrix Sigma, array[] int new_order, int N){
        // reorders corr matrix to follow new_order
        matrix[N,N] new_Sigma;
        for (i in 1:N){
            for (j in 1:N){
                new_Sigma[i][j] = Sigma[new_order[i]][new_order[j]];
            }
        }
        return new_Sigma;
    }

    vector reorder_vector(vector to_reorder, array[] int new_order){
        // reorders vector to follow new_order
        int N = dims(new_order)[1];
        vector[N] new_vector;
        for (i in 1:N){
            new_vector[i] = to_reorder[new_order[i]];
        }
        return new_vector;
    }

    // unusable func: can't return multiple matrices of different sizes
    // array[] matrix create_cond_gaussian(matrix Sigma, int len_x1, int len_x2, vector Z) {
    //     // see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    //     // len_x1 is the number of elements in the first (unobserved) section
    //     // len_x2 is the number of elements in the second (observed) section
    //     // Z is the observed data, should have length len_x2
    //     vector[len_x1] mu_1 = rep_vector(0, len_x1);
    //     vector[len_x2] mu_2 = rep_vector(0, len_x2);

    //     matrix[len_x1, len_x1] Sigma_11 = block(Sigma, 1, 1, len_x1, len_x1);
    //     matrix[len_x1, len_x2] Sigma_12 = block(Sigma, 1, len_x1 + 1, len_x1, len_x2);
    //     matrix[len_x2, len_x1] Sigma_21 = block(Sigma, len_x1 + 1, 1, len_x2, len_x1);
    //     matrix[len_x2, len_x2] Sigma_22 = block(Sigma, len_x1 + 1, len_x1 + 1, len_x2, len_x2);

    //     matrix[len_x1, len_x2] m_prod = mdivide_right_spd(Sigma_12, Sigma_22);

    //     matrix[len_x1,1] mu_cond = mu_1 + m_prod * (Z - mu_2);

    //     matrix[len_x1, len_x1] Sigma_cond;
    //     Sigma_cond = Sigma_11 - m_prod * Sigma_21;
        
    //     array[2] matrix return_arr;
    //     return_arr = [mu_cond, Sigma_cond];
    //     return return_arr;
    // }

    matrix create_marginal_gaussian(matrix Sigma, array[] int include_ind, int N) {
        int new_N = dims(include_ind)[1];
        matrix[new_N, new_N] Sigma_marg;
        int i_ind = 1;
        for (i in 1:N) {
            int j_ind = 1;
            if (i_in(i, include_ind)) {
                for (j in 1:N) {
                    if (i_in(j, include_ind)) {
                        Sigma_marg[i_ind][j_ind] = Sigma[i][j];
                        j_ind += 1;
                    }
                }
                i_ind += 1;
            }
        }
        return Sigma_marg;
    }
}
data {

    // if zero, just simulate z_U^H + convert to prob vector
    // if one, simulate \hat{y}
    int<lower=0, upper=1> simulate_y;

    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> K; 

    matrix[(K-1)*(n_models + n_humans),(K-1)*(n_models + n_humans)] Sigma;

    // need at least one [hypothetical] human so we have a z_H to estimate
    int<lower=1> n_observed_humans;

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

    int n_observed = (K-1)*(n_models+n_observed_humans);
    int n_unobserved = N - n_observed; 
    int n_unobserved_humans = n_humans - n_observed_humans;

    // arrays of true indices
    array[n_models] int model_ind;
    array[n_observed_humans] int observed_true_ind;
    array[n_unobserved_humans] int unobserved_true_ind;
    array[n_models+n_observed_humans] int observed_ind_full;

    for (i in 1:n_models) {
        model_ind[i] = i;
    }
    int u_ind = 1;
    int o_ind = 1;
    for (i in 1:n_humans) {
        if (i_in(i, unobserved_ind)) {
            unobserved_true_ind[u_ind] = n_models + i;
            u_ind += 1;
        } else {
            observed_true_ind[o_ind] = n_models + i;
            o_ind += 1;
        }
    }

    observed_ind_full = append_array(model_ind, observed_true_ind);

    // get mu and Sigma for marginal Gaussian on z_H^O
    int observed_N = n_observed_humans*(K-1);
    array[observed_N] int include_ind = expand_to_k(observed_true_ind, K);
    matrix[observed_N,observed_N] Sigma_sub = create_marginal_gaussian(Sigma, include_ind, N);
    vector[observed_N] mu_sub = rep_vector(0, observed_N);

    // create conditional Gaussian for Z_M given Z_H^O
    int sub_size = (n_models + n_observed_humans)*(K-1);
    // put unobserved humans at the end since reorder_Sigma reorders full matrix
    array[n_models + n_humans] int new_order = append_array(append_array(observed_true_ind, model_ind), unobserved_true_ind); 
    array[N] int new_order_ind = expand_to_k(new_order, K);
    // reorder Sigma, then trim off unobserved humans
    matrix[N,N] Sigma_cr_full = reorder_Sigma(Sigma, new_order_ind, N);
    matrix[sub_size,sub_size] Sigma_cr = block(Sigma_cr_full, 1, 1, sub_size, sub_size);
    
    int s1 = n_observed_humans*(K-1); // first block: unobserved (to estimate)
    int s2 = n_models*(K-1); // second block: observed (to condition on)

    vector[s1] mu_1 = rep_vector(0, s1);
    vector[s2] mu_2 = rep_vector(0, s2);

    matrix[s1, s1] Sigma_11 = block(Sigma_cr, 1, 1, s1, s1);
    matrix[s1, s2] Sigma_12 = block(Sigma_cr, 1, s1 + 1, s1, s2);
    matrix[s2, s1] Sigma_21 = block(Sigma_cr, s1 + 1, 1, s2, s1);
    matrix[s2, s2] Sigma_22 = block(Sigma_cr, s1 + 1, s1 + 1, s2, s2);

    matrix[s1, s2] m_prod = mdivide_right_spd(Sigma_12, Sigma_22);
    matrix[s1, s1] Sigma_c;
    Sigma_c = Sigma_11 - m_prod * Sigma_21;


    // for conditional Gaussian Z_H^U | Z_M, Z_H^O
    int s1_g = n_unobserved_humans*(K-1); // first block: unobserved (to estimate)
    int s2_g = (n_models+n_observed_humans)*(K-1); // second block: observed (to condition on)
    // for reordering Sigma for Z_H^U | Z_M, Z_H^O
    array[n_models+n_humans] int new_order_g = append_array(unobserved_true_ind, observed_ind_full);
    array[N] int new_order_ind_g = expand_to_k(new_order_g, K);

}
parameters { 
    vector[n_observed_humans*(K-1)] Z_H_new;
}
transformed parameters {
    // mu for conditional Gaussian for Z_M given Z_H^O
    vector[s1] mu_c = mu_1 + m_prod * (Z_H_new - mu_2);
}
model {
    // set priors for z_H: marginal gaussian
    Z_H_new ~ multi_normal(mu_sub, Sigma_sub);

    // categorical for Y_H given Z_H
    for (i in 1:n_observed_humans) {
        array[K-1] real Z_H_new_i = to_array_1d(segment(Z_H_new, (K-1)*(i-1)+1, K-1));
        vector[K] Pmf = to_vector(additive_logistic(Z_H_new_i)); 
        target += categorical_lpmf( Y_H_new[i] | Pmf);
    }

    // gaussian for Z_M conditional on Z_H
    target += multi_normal_lpdf(Z_M | mu_c, Sigma_c);
}
generated quantities {

    vector[n_unobserved_humans*(K-1)] Z_H_new_sample;

    {
        // reorder Sigma for cond. Gaussian Z_H^U given Z_M, Z_H^O
        matrix[N,N] Sigma_r = reorder_Sigma(Sigma, new_order_ind_g, N);

        // create Z
        // new_order_ind_g is unobserved humans, models, then observed humans
        // so don't need to reorder Z
        vector[(n_models + n_observed_humans)*(K-1)] Z = append_row(Z_M, Z_H_new);

        // create conditional Gaussian parameters Z_H^U | Z_M, Z_H^O
        vector[s1_g] mu_1_g = rep_vector(0, s1_g);
        vector[s2_g] mu_2_g = rep_vector(0, s2_g);

        matrix[s1_g, s1_g] Sigma_11_g = block(Sigma_r, 1, 1, s1_g, s1_g);
        matrix[s1_g, s2_g] Sigma_12_g = block(Sigma_r, 1, s1_g + 1, s1_g, s2_g);
        matrix[s2_g, s1_g] Sigma_21_g = block(Sigma_r, s1_g + 1, 1, s2_g, s1_g);
        matrix[s2_g, s2_g] Sigma_22_g = block(Sigma_r, s1_g + 1, s1_g + 1, s2_g, s2_g);

        matrix[s1_g, s2_g] m_prod_g = mdivide_right_spd(Sigma_12_g, Sigma_22_g);
        matrix[s1_g, s1_g] Sigma_cond;
        Sigma_cond = Sigma_11_g - m_prod_g * Sigma_21_g;

        vector[s2_g] mu_cond = mu_1_g + m_prod_g * (Z - mu_2_g);
        Z_H_new_sample = multi_normal_rng(mu_cond, Sigma_cond);
    }

    // ESTIMATE P(Y) VIA SAMPLING

    //vector[n_unobserved_humans*(K-1)] Z_H_new_sample;
    //Z_H_new_sample = multi_normal_rng(mu_cond, Sigma_cond);

    vector[K] p_y = rep_vector(0, K);
    matrix[n_unobserved_humans,K] latent_probs;

    if (simulate_y) {

        array[n_draws] int y_samples;

        for (d in 1:n_draws) {
            array[K] int votes = rep_array(0, K);
            for (i in 1:n_observed_humans) {
                votes[Y_H_new[i]] += 1;
            }
            for (i in 1:n_unobserved_humans) {
                // segment of Z_H corresponding to person i
                array[K-1] real Z_H_i;
                Z_H_i = to_array_1d(segment(Z_H_new_sample, (K-1)*(i-1)+1, K-1));
                vector[K] Pmf;
                Pmf = to_vector(additive_logistic(Z_H_i));
                int vote_i = categorical_rng(Pmf);
                votes[vote_i] += 1;
            }
            int y_sample = argmax(votes);
            y_samples[d] = y_sample;
        }

        for (d in 1:n_draws) {
            p_y[y_samples[d]] += 1;
        }
        p_y = p_y ./ n_draws;

    } else {
        // save z_U^H
        for (i in 1:n_unobserved_humans) {
            // segment of Z_H corresponding to person i
            array[K-1] real Z_H_i;
            Z_H_i = to_array_1d(segment(Z_H_new_sample, (K-1)*(i-1)+1, K-1));
            vector[K] Pmf;
            latent_probs[i] = to_row_vector(additive_logistic(Z_H_i));
        }
    }


    // COMPUTE PROBABILITY OF CORRECT PREDICTION FOR EACH CANDIDATE
    
    // array[n_humans - n_observed_humans] real p_i_correct;
    // for (i in unobserved_ind) {
    //     real p_correct = 0;
    //     array[K-1] real Z_H_i;
    //     Z_H_i = to_array_1d(segment(Z_H_new, (K-1)*(i-1)+1, K-1));
    //     array[K] real Z_H_i_trans;
    //     Z_H_i_trans = additive_logistic(Z_H_i);
    //     for (k in 1:K) {
    //         p_correct += Z_H_i_trans[k] * p_y[k];
    //     }
    //     p_i_correct[i] = p_correct;
    // }

}