import shutil
import time

import numpy as np
from scipy.stats import entropy

from utils import get_consensus, print_Sigma

class ConsensusModel:
    '''
    A model for predicting consensus for a particular dataset.
    '''

    def __init__(
        self, 
        dataset, 
        mvn_fit,
        stan_model,
        model_id
    ):
        '''
        `dataset`: instance of a `Dataset`
        `mvn_fit`: stan fit with estimates of underlying normal
            parameters, result of update_parameters.stan
        `stan_model`: CmdStanModel instance for consensus prediction
            (simulate_consensus.stan)
        `model_id`: unique ID for this model run (to prevent conflicts
            with stan temporary files)
        '''
        self.dataset = dataset
        self.mvn_fit = mvn_fit
        self.stan_model = stan_model
        self.id = model_id
        

    def model_consensus(self, stan_dict, id_str):
        '''
        fit a consensus model for the data in `stan_dict` in a unique
        temporary directory with ending `id_str`. `out_dir` should be deleted
        after needed quantities are extracted from `fit`
        '''
        timestamp = str(time.time()).split(".")
        out_dir = "tmp/" + timestamp[0] + self.id + id_str
        fit = self.stan_model.generate_quantities(
            data=stan_dict, 
            previous_fit=self.mvn_fit,
            gq_output_dir=out_dir
        )
        return fit, out_dir

    
    def consensus_dist(self, stan_dict, id_str):
        '''
        compute and return the (normalized) distribution over the consensus y
        for the data in `stan_dict` using `id_str` for the unique temporary
        directory.
        '''
        fit, out_dir = self.model_consensus(stan_dict, id_str)
        consensus_dist_est = np.zeros(self.dataset.K)
        for m in range(self.dataset.K):
            consensus_dist_est[m] = fit.stan_variable("p_y")[:,m].mean()
        shutil.rmtree(out_dir)
        consensus_dist_norm = consensus_dist_est/sum(consensus_dist_est)

        return consensus_dist_norm


    def get_expert_choice_probabilities(self, example):
        '''
        compute and return a (# of unobserved experts)*K matrix for `example`,
        where entry i,j corresponds to the estimated probability that the ith
        unobserved expert will choose class j
        '''
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
        '''
        for a given `example`, for each unobserved expert, compute the expected
        entropy of the consensus distribution after observing their vote and
        return the index of the expert with the minimum expected entropy
        '''

        candidate_ees = {}

        choice_probabilities = self.get_expert_choice_probabilities(example)
        c_index = 0

        for c in example.unobserved_ind:

            expected_entropy = 0

            for j in range(1, self.dataset.K+1):

                hyp_dict = example.get_hypothetical_stan_dict(c, j)

                consensus_dist_est = self.consensus_dist(
                    stan_dict=hyp_dict,
                    id_str = str(c) + str(j)
                )

                #probability candidate c chooses class j
                p_k = choice_probabilities[c_index][j-1]

                e = entropy(consensus_dist_est)
                expected_entropy += p_k * e
            
            c_index +=1
            candidate_ees[c] = expected_entropy

        chosen_expert = min(candidate_ees, key=candidate_ees.get)
        return chosen_expert


    def query_next_human(self, example):
        '''
        query the unobserved expert that minimizes expected entropy and
        update the state of `example`
        '''
    
        if len(example.unobserved_ind)>1:
            chosen_expert = self.choose_expert(example)
        else:
            chosen_expert = example.unobserved_ind[0]

        # query chosen expert & update dict
        example.update_with_query(chosen_expert)


    def get_prediction(self, i, threshold):
        '''
        get the consensus prediction for test example `i`, querying human
        experts until the % uncertainty falls below `threshold` (in [0,1]).
        update `self.dataset` with the new observation
        '''

        # create an `Example` for the test example at index `i` 
        example = self.dataset.get_test_example(i)
        true_consensus = self.dataset.get_human_consensus(i)
        num_queries = 0

        consensus_dist = self.consensus_dist(example.get_stan_dict(), 'cdm')
        uncertainty = 1 - max(consensus_dist)

        while uncertainty > threshold:
            # query new expert
            num_queries += 1
            self.query_next_human(example)
            consensus_dist = self.consensus_dist(example.get_stan_dict(), 'cd')
            uncertainty = 1 - max(consensus_dist)

        self.dataset.update(example)

        pred_y = np.argmax(consensus_dist) + 1
        result = {
                'data_index' : i,
                'n_queries' : num_queries,
                'pred_y' : pred_y,
                'uncertainty' : uncertainty,
                'correct' : pred_y == true_consensus
        }
        return result


    def get_prediction_random_querying(self, i, threshold):
        '''
        get the consensus prediction for test example `i`, querying until
        uncertainty falls below `threshold`, by randomly querying experts
        '''

        example = self.dataset.get_test_example(i)
        true_consensus = self.dataset.get_human_consensus(i)
        num_queries = 0

        consensus_dist = self.consensus_dist(example.get_stan_dict(), 'cdm')
        uncertainty = 1 - max(consensus_dist)

        while uncertainty > threshold:
            # query random expert
            num_queries += 1
            random_expert = np.random.choice(example.unobserved_ind)
            stan_dict = example.update_with_query(random_expert)
            consensus_dist = self.consensus_dist(stan_dict, 'cd')
            uncertainty = 1 - max(consensus_dist)

        self.dataset.update(example)
        
        pred_y = np.argmax(consensus_dist) + 1
        result = {
                'data_index' : i,
                'n_queries' : num_queries,
                'pred_y' : pred_y,
                'uncertainty' : uncertainty,
                'correct' : pred_y == true_consensus
        }
        return result


    def get_prediction_simple_consensus(self, i, n_queries):
        '''
        predict consensus for example `i`, querying `n_queries` human
        experts and using simple consensus to predict y (instead of the model)
        '''

        example = self.dataset.get_test_example(i)
        true_consensus = self.dataset.get_human_consensus(i)

        for q in range(n_queries):
            num_queries += 1
            self.query_next_human(example)

        if n_queries == 0:
            pred_y_options = [i for i in range(1,self.dataset.K+1)]
        else:
            _, pred_y_options = get_consensus(example.Y_O)

        self.dataset.update(example)
        
        pred_y = np.random.choice(pred_y_options)

        result = {
                'data_index' : i,
                'n_queries' : num_queries,
                'pred_y' : pred_y,
                'correct' : pred_y == true_consensus
        }
        return result
