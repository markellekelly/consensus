from bisect import bisect
import abc
import numpy as np
from utils import get_consensus


class Example(abc.ABC):
    '''
    Corresponds to a specific data instance. Keeps track of which experts have 
    already been queried. To use, implement `query_expert`.
    '''

    def __init__(
        self,
        base_dict,
        n_humans,
        model_predictions
    ):
        self.base_dict = base_dict
        self.Y_M = model_predictions
        self.Y_O = []
        self.n_humans = n_humans
        self.n_observed_humans = 0
        self.unobserved_ind = [i for i in range(1, self.n_humans+1)]

    def get_stan_dict(self):
        '''
        get a dictionary that matches the needed stan data input
        reflects the current state (i.e., which experts have been queried)
        '''
        test_dict = {
            'Y_M_new' : self.Y_M,
            'Y_O_real' : self.Y_O,
            'n_observed_humans' : self.n_observed_humans,
            'unobserved_ind' : self.unobserved_ind,
        }
        test_dict.update(self.base_dict)
        return test_dict

    def process_new_vote(self, human_index, vote):
        '''
        return the new lists of unobserved human indicies and observed human
        votes that would result from the `human_index` expert voting `vote`
        '''
        unobserved_ind = [
            j for j in self.unobserved_ind if j != human_index
        ]
        observed_ind = [
            j for j in range(1, self.n_humans+1) if j not in unobserved_ind
        ]
        insert_at = bisect(observed_ind, human_index-1)
        Y_O_new = self.Y_O.copy()
        Y_O_new.insert(insert_at, vote)
        return unobserved_ind, Y_O_new

    @abc.abstractmethod
    def query_expert(self, human_index):
        '''
        get the vote of the expert at `human_index` (integer between 1,... K.)
        '''
        pass

    def update_with_query(self, human_index):
        '''
        query the expert at `human_index`, update internal state, and 
        return the updated data dictionary for stan
        '''
        self.n_observed_humans += 1
        query_result = self.query_expert(human_index)
        self.unobserved_ind, self.Y_O = self.process_new_vote(
            human_index, query_result
        )
        return self.get_stan_dict()

    def get_hypothetical_stan_dict(self, human_index, vote):
        '''
        get stan data dictionary corresponding to observing a hypothetical 
        expert/vote combination, without updating internal state
        '''
        unobserved_ind, Y_O_new = self.process_new_vote(human_index, vote)
        hypothetical_dict = self.get_stan_dict().copy()
        hypothetical_dict['n_observed_humans'] += 1
        hypothetical_dict['unobserved_ind'] = unobserved_ind
        hypothetical_dict['Y_O_real'] = Y_O_new
        return hypothetical_dict


class TestExample(Example):
    '''
    A test example with known model and human expert predictions.
    '''

    def __init__(
        self,
        base_dict,
        n_humans,
        model_predictions,
        human_predictions
    ):
        super().__init__(base_dict, n_humans, model_predictions)
        self.Y_H = human_predictions

    def query_expert(self, human_index):
        '''
        get the vote of a the expert at `human_index`
        '''
        # in our experiments, we already have all human votes 
        return self.Y_H[human_index-1]


class Dataset:

    def __init__(
        self, 
        n_models, 
        n_humans, 
        n_initialization_items,
        n_classes,
        model_predictions,
        human_predictions
    ):
        self.n_models = n_models
        self.n_humans = n_humans
        self.n_items = n_initialization_items
        self.K = n_classes
        self.Y_M = model_predictions[:n_initialization_items]
        self.Y_H = human_predictions[:n_initialization_items]
        self.Y_M_new = model_predictions[n_initialization_items:]
        self.Y_H_new = human_predictions[n_initialization_items:]
        self.base_dict = self.get_base_stan_dict()
        self.n = (self.n_models + self.n_humans)*(self.K - 1)

    def get_human_consensus(self, i):
        human_labels = self.Y_H_new[i]
        consensus, _ = get_consensus(human_labels)
        return consensus

    def get_model_prediction(self, i):
        return np.argmax(self.Y_M_new[i]) + 1

    def get_base_stan_dict(self):
        return {
            'n_models' : self.n_models,
            'n_humans' : self.n_humans,
            'n_items' : self.n_items,
            'K' : self.K,
            'use_temp_scaling' : 1
        }

    def get_init_stan_dict(self, eta=0.75):
        data_dict = {
            'Y_M' : self.Y_M,
            'Y_H' : self.Y_H,
            'eta' : eta,
        }
        data_dict.update(self.base_dict)
        return data_dict

    def get_test_example(self, ind):
        return TestExample(
            base_dict = self.base_dict, 
            n_humans = self.n_humans,
            model_predictions = self.Y_M_new[ind],
            human_predictions = self.Y_H_new[ind]
        )
        
