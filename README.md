# consensus

- **model.stan** estimates the correlation matrix Sigma for Z (which contains Z_M, the transformed model class-wise probabilities, and Z_H, the transformed latent human class_wise probabilities).
- **expert_inference.stan** computes the probabilities that each unobserved human's prediction will be correct (i.e., agree with the consensus), given already-observed model outputs Y_M_new and, if applicable, the already-observed human predictions Y_H_new. A simple heuristic for choosing the next expert is to choose the argmax of p_i_correct.
- **test_full_model.py** contains an end-to-end workflow of the use of both stan models on a simple simulated example.