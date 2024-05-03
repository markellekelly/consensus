import shutil
import time

import numpy as np
from scipy.stats import entropy

from utils import get_consensus

class ConsensusModel:

    def __init__(
        self, 
        dataset, 
        mvn_fit,
        stan_model,
        model_id
    ):
        self.dataset = dataset
        self.mvn_fit = mvn_fit
        self.stan_model = stan_model
        self.id = model_id
        

    def model_consensus(self, stan_dict, id_str):
        timestamp = str(time.time()).split(".")
        out_dir = "tmp/" + timestamp[0] + self.id + id_str
        fit = self.stan_model.generate_quantities(
            data=stan_dict, 
            previous_fit=self.mvn_fit,
            gq_output_dir=out_dir
        )
        return fit, out_dir


    def get_expert_choice_probabilities(self, example):
        candidates = example.unobserved_ind
        current_fit, out_dir = self.model_consensus(
            example.get_stan_dict(), 
            "-ec"
        )
        likelihood = current_fit.stan_variable("p_y_k")
        votes = [
            current_fit.stan_variable("Y_U")[:,e] for e in range(len(candidates))
        ]
        shutil.rmtree(out_dir)

        num_votes = len(votes[0])
        probabilities_un = np.zeros((len(candidates), self.dataset.K))
        c_index = 0
        for c in candidates:
            for j in range(1, self.dataset.K+1):
                sample_count = [1 if int(v)==j else 0 for v in votes[c_index]]
                prob_cj = sum(sample_count * likelihood)/num_votes
                probabilities_un[c_index][j-1] = prob_cj
            c_index+=1

        # normalize to get probabilities
        row_sums = probabilities_un.sum(axis=1, keepdims=True)
        probabilities = probabilities_un / row_sums

        return probabilities

    
    def choose_expert(self, example):

        candidate_ees = {}

        choice_probabilities = self.get_expert_choice_probabilities(example)
        c_index = 0

        for c in example.unobserved_ind:

            print('assessing candidate {}...'.format(c))
            expected_entropy = 0

            for j in range(1, self.dataset.K+1):

                print('if they vote {}...'.format(j))
                hyp_dict = example.get_hypothetical_stan_dict(c, j)
                print(hyp_dict)

                consensus_dist_est = self.consensus_dist(
                    stan_dict=hyp_dict,
                    id_str = str(c) + str(j)
                )

                #probability candidate c chooses class j
                p_k = choice_probabilities[c_index][j-1]
                print("{}% chance".format(p_k))

                e = entropy(consensus_dist_est)
                expected_entropy += p_k * e
            
            c_index +=1
            candidate_ees[c] = expected_entropy

        chosen_expert = min(candidate_ees, key=candidate_ees.get)
        print("candidate ees:" + str(candidate_ees))
        print("chose "+ str(chosen_expert))
        return chosen_expert


    def consensus_dist(self, stan_dict, id_str):

        fit, out_dir = self.model_consensus(stan_dict, id_str)
        consensus_dist_est = np.zeros(self.dataset.K)
        for m in range(self.dataset.K):
            consensus_dist_est[m] = fit.stan_variable("p_y")[:,m].mean()
        shutil.rmtree(out_dir)
        consensus_dist_norm = consensus_dist_est/sum(consensus_dist_est)

        return consensus_dist_norm


    def query_next_human(self, example):
    
        if len(example.unobserved_ind)>1:
            chosen_expert = self.choose_expert(example)
        else:
            chosen_expert = example.unobserved_ind[0]

        # query chosen expert & update dict
        example.update_with_query(chosen_expert)


    def get_prediction(self, row_index, threshold):

        example = self.dataset.get_test_example(row_index)

        consensus_dist = self.consensus_dist(example.get_stan_dict(), 'cdm')
        uncertainty = 1 - max(consensus_dist)

        while uncertainty > threshold:
            # query new expert
            self.query_next_human(example)
            consensus_dist = self.consensus_dist(example.get_stan_dict(), 'cd')
            uncertainty = 1 - max(consensus_dist)

        pred_y = np.argmax(consensus_dist) + 1
        return pred_y
