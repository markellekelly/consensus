from bisect import bisect
import abc
import numpy as np
from utils import get_consensus


class Example(abc.ABC):
    '''
    Corresponds to one input x^(t), keeping track of which experts have 
    already been queried for that example. `query_expert` is abstract.
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
            'eta' : self.base_dict["eta"],
            'n_observed_humans' : self.n_observed_humans,
            'unobserved_ind' : self.unobserved_ind,
        }
        test_dict.update(self.base_dict)
        return test_dict

    def get_Y_H(self):
        '''
        return a list of observed human votes, using 0 for unobserved
        '''
        Y_H = []
        current_ind = 0
        for i in range(1, self.n_humans+1):
            if i in self.unobserved_ind:
                Y_H.append(0)
            else:
                Y_H.append(self.Y_O[current_ind])
                current_ind += 1
        return Y_H

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
        get the vote of the `human_index`th expert (integer between 1,... K.)
        input: int `human_index` in {1,..., `self.n_humans`}
        returns: int vote in {1, ... K}
        '''
        pass

    def update_with_query(self, human_index):
        '''
        query the expert at `human_index`, update internal state, and 
        return the updated data dictionary for stan
        '''
        self.n_observed_humans += 1
        query_result = self.query_expert(human_index)
        # update status of observed and unobserved experts
        self.unobserved_ind, self.Y_O = self.process_new_vote(
            human_index, query_result
        )
        return self.get_stan_dict()

    def get_hypothetical_stan_dict(self, human_index, vote):
        '''
        get stan data dictionary corresponding to observing a hypothetical 
        new expert/vote combination, without updating internal state
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


class Dataset(abc.ABC):
    '''
    A class corresponding to a dataset, including an initialization set
    with model and human predictions.
    '''

    def __init__(
        self, 
        n_models, 
        n_humans, 
        n_classes,
        model_predictions = [],
        human_predictions= [],
        use_temp_scaling = 1, 
        use_correlations = 1,
        eta = 0.75
    ):
        '''
        `model_predictions`: list of lists of model predictions with shape
            (total # of examples)*`n_models`*`n_classes`
        `human_predictions`: list of human predictions with shape
            (total # of examples)*n_humans, values in {1, ..., `n_classes`}
        `use_temp_scaling`: int (0 or 1) that controls whether the human
            predictions should be calibrated via temperature scaling
        `eta`: value for the eta hyperparameter 
        '''
        self.n_models = n_models
        self.n_humans = n_humans
        self.n_items = len(model_predictions)
        self.K = n_classes
        self.Y_M = model_predictions
        self.Y_H = human_predictions
        self.eta = eta
        self.use_temp_scaling = use_temp_scaling
        self.use_correlations = use_correlations
        self.base_dict = self.get_base_stan_dict()
        self.n = (self.n_models + self.n_humans)*(self.K - 1)

    def get_base_stan_dict(self):
        '''
        return a dictionary containing data needed for both the initialization
        and consensus prediction models (in stan)
        '''
        return {
            'n_models' : self.n_models,
            'n_humans' : self.n_humans,
            'n_items' : self.n_items,
            'eta' : self.eta,
            'K' : self.K,
            'use_temp_scaling' : self.use_temp_scaling,
            'use_correlations' : self.use_correlations
        }

    def get_init_stan_dict(self):
        '''
        return a dictionary with the needed inputs for the underlying_normal
        stan model, including hyperparameter `eta`
        '''
        data_dict = { 'Y_M' : self.Y_M, 'Y_H' : self.Y_H }
        data_dict.update(self.base_dict)
        return data_dict

    def update(self, example):
        '''
        update the observed dataset with a new example with model predictions
        `Y_M_observed` and (potentially partial) expert votes `Y_H_observed`
        '''
        self.n_items += 1
        self.Y_M.append(example.Y_M)
        self.Y_H.append(example.get_Y_H())
        self.base_dict = self.get_base_stan_dict()

    @abc.abstractmethod
    def get_test_example(self, i):
        '''
        return an instance of an `Example` corresponding to index `i`
        '''


class TestDataset(Dataset):
    '''
    A class corresponding to a dataset with known human and model predictions
    for some test set `model_predictions_test` and `human_predictions_test`
    '''

    def __init__(self, model_predictions_test, human_predictions_test, **args):
        self.Y_M_new = model_predictions_test
        self.Y_H_new = human_predictions_test
        super().__init__(**args)

    def get_human_consensus(self, i):
        '''
        return the expert consensus for test example `i`
        '''
        human_labels = self.Y_H_new[i]
        consensus, _ = get_consensus(human_labels)
        return consensus

    def get_model_prediction(self, i):
        '''
        return the model prediction for test example `i`
        '''
        return np.argmax(self.Y_M_new[i]) + 1

    def update(self, example):
        '''
        move the first `n_examples` examples from the test set to our observed
        set, given the partially observed expert votes `Y_H_observed`
        '''
        self.Y_M_new = self.Y_M_new[1:]
        self.Y_H_new = self.Y_H_new[1:]
        super().update(example)

    def get_test_example(self, i):
        '''
        create & return a `TestExample` corresponding to test example `i`
        '''

        return TestExample(
            base_dict = self.base_dict, 
            n_humans = self.n_humans,
            model_predictions = self.Y_M_new[i],
            human_predictions = self.Y_H_new[i]
        )
        
