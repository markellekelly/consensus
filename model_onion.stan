
functions {

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

    matrix onion(vector R2, vector l, int N) {
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

    int<lower=1> n_models;
    int<lower=1> n_humans;
    int<lower=1> n_items;
    int<lower=1> K; 

    // model predictions: probabilities (\in a k-simplex)
    array[n_items,n_models,K] real<lower=0,upper=1> Y_M;

    // human predictions: votes (\in {1, ... K})
    array[n_items,n_humans] int<lower=1,upper=K> Y_H;

    real<lower = 0> eta;

} 
transformed data {

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

    int N = (K-1)*(n_models + n_humans);

    real<lower = 0> alpha = eta + (N - 2) / 2.0;
    vector<lower = 0>[N-1] shape1;
    vector<lower = 0>[N-1] shape2;

    shape1[1] = alpha;
    shape2[1] = alpha;
    for(n in 2:(N-1)) {
        alpha = alpha - 0.5;
        shape1[n] = n / 2.0;
        shape2[n] = alpha;
  }

}
parameters { 

    vector[choose(N, 2) - 1]  l;         // do NOT init with 0 for all elements
    vector<lower = 0,upper = 1>[N-1] R2; // first element is not really a R^2 but is on (0,1)  
    
    // see https://mc-stan.org/docs/stan-users-guide/partially-known-parameters.html
    matrix[n_items,n_humans*(K-1)] Z_H;

}  
transformed parameters {
    
    matrix[n_items, N] Z;
    for (i in 1:n_items){
        for (j in 1:n_models*(K-1)){
            Z[i][j] = Z_M[i][j];
        }
        for (k in 1:n_humans*(K-1)){
            Z[i][k+(n_models*(K-1))] = Z_H[i][k];
        }
    }

    matrix[N, N] L_Sigma = onion(R2, l, N);

}
model {

    l ~ normal(0.0, 1.0);
    R2 ~ beta(shape1, shape2);
    
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
            int ind = (K-1) * (n_models + j - 1);
            for (k in 1:K-1){
                to_transform[k] = Z[i][ind+k];
            }
            Pmf = to_vector(additive_logistic(to_transform)); 
            Y_H[i,j] ~ categorical(Pmf);
        }
    }

}
generated quantities {

    matrix[N,N] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);

    // matrix[N,N] Omega;
    // Omega = multiply_lower_tri_self_transpose(L_Omega);

}