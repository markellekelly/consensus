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

    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> K; 

    matrix[(K-1)*(n_models + n_humans),(K-1)*(n_models + n_humans)] Sigma;

    // model predictions: probabilities (\in a k-simplex)
    array[n_models,K] real<lower=0,upper=1> Y_M_new;
} 
transformed data {

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

    int n_observed = (K-1)*(n_models);
    int n_unobserved = N - n_observed; 

    // arrays of true indices
    array[n_models] int model_ind;
    array[n_humans] int human_ind;

    for (i in 1:n_models) {
        model_ind[i] = i;
    }
    for (i in 1:n_humans) {
        human_ind[i] = n_models + i;
    }

    // get mu and Sigma for marginal Gaussian on z_H
    int h_N = n_humans*(K-1);
    array[h_N] int include_ind = expand_to_k(human_ind, K);
    matrix[h_N,h_N] Sigma_sub = create_marginal_gaussian(Sigma, include_ind, N);
    vector[h_N] mu_sub = rep_vector(0, h_N);

    // create conditional Gaussian for Z_M given Z_H
    array[n_models + n_humans] int new_order = append_array(human_ind, model_ind); 
    array[N] int new_order_ind = expand_to_k(new_order, K);
    // reorder Sigma
    matrix[N,N] Sigma_cr = reorder_Sigma(Sigma, new_order_ind, N);
    
    int s1 = n_models*(K-1); // first block: unobserved (to estimate)
    int s2 = n_humans*(K-1); // second block: observed (to condition on)

    vector[s1] mu_1 = rep_vector(0, s1);
    vector[s2] mu_2 = rep_vector(0, s2);

    matrix[s1, s1] Sigma_11 = block(Sigma_cr, 1, 1, s1, s1);
    matrix[s1, s2] Sigma_12 = block(Sigma_cr, 1, s1 + 1, s1, s2);
    matrix[s2, s1] Sigma_21 = block(Sigma_cr, s1 + 1, 1, s2, s1);
    matrix[s2, s2] Sigma_22 = block(Sigma_cr, s1 + 1, s1 + 1, s2, s2);

    matrix[s1, s2] m_prod = mdivide_right_spd(Sigma_12, Sigma_22);
    matrix[s1, s1] Sigma_c;
    Sigma_c = Sigma_11 - m_prod * Sigma_21;

}
parameters { 
    vector[n_humans*(K-1)] Z_H;
}
transformed parameters {
    // mu for conditional Gaussian for Z_M given Z_H
    vector[s1] mu_c = mu_1 + m_prod * (Z_H - mu_2);
}
model {
    // set priors for z_H: marginal gaussian
    Z_H ~ multi_normal(mu_sub, Sigma_sub);

    // gaussian for Z_M conditional on Z_H
    target += multi_normal_lpdf(Z_M | mu_c, Sigma_c);
}
generated quantities {

    matrix[n_humans,K] latent_probs;
    for (i in 1:n_humans) {
        // segment of Z_H corresponding to person i
        array[K-1] real Z_H_i;
        Z_H_i = to_array_1d(segment(Z_H, (K-1)*(i-1)+1, K-1));
        vector[K] Pmf;
        latent_probs[i] = to_row_vector(additive_logistic(Z_H_i));
    }

}