from bisect import bisect
import numpy as np
from utils import get_consensus

class Dataset:

    def __init__(
        self, 
        n_models, 
        n_humans, 
        n_initialization_items,
        n_classes,
        model_predictions,
        human_predictions,
        apply_temperature_scaling
    ):
        self.n_models = n_models
        self.n_humans = n_humans
        self.n_items = n_initialization_items
        self.K = n_classes
        self.Y_M = model_predictions[:n_initialization_items]
        self.Y_H = human_predictions[:n_initialization_items]
        self.Y_M_new = model_predictions[n_initialization_items:]
        self.Y_H_new = human_predictions[n_initialization_items:]
        self.apply_ts = apply_temperature_scaling
        self.test_dict = {}
        self.hyp_dict = {}
        self.i = 0

    def get_human_consensus(self, i):
        human_labels = self.Y_H_new[i]
        consensus, _ = get_consensus(human_labels)
        return consensus

    def get_model_prediction(self, i):
        return np.argmax(self.Y_M_new[i]) + 1

    def get_init_data_dict(self, eta=0.75):
        return {
            'n_models' : self.n_models,
            'n_humans' : self.n_humans,
            'n_items' : self.n_items,
            'K' : self.K,
            'Y_M' : self.Y_M,
            'Y_H' : self.Y_H,
            'eta' : eta,
            'use_temp_scaling' : int(self.apply_ts)
        }

    def init_test(self, i):
        self.i = i
        self.observed_ind = []
        self.test_dict = {
            'n_models' : self.n_models,
            'n_humans' : self.n_humans,
            'n_items' : self.n_items,
            'K' : self.K,
            'Y_M_new' : self.Y_M_new[i],
            'Y_O_real' : [],
            'n_observed_humans' : 0,
            'unobserved_ind' : [i for i in range(1, self.n_humans+1)],
            'use_temp_scaling' : int(self.apply_ts)
        }
        self.hyp_dict = self.test_dict.copy()

    def get_test_dict(self):
        return self.test_dict

    def update_dict_with_vote(self, u_dict, human_index, vote=None):
        u_dict["n_observed_humans"] += 1
        u_dict["unobserved_ind"] = [
            j for j in u_dict["unobserved_ind"] if j != human_index
        ]
        observed_ind = [
            j for j in range(1, self.n_humans+1) if j not in u_dict["unobserved_ind"]
        ]
        insert_at = bisect(observed_ind, human_index-1)
        if vote:
            query_result = vote
        else:
            query_result = self.Y_H_new[self.i][human_index-1]
        Y_O_new = u_dict["Y_O_real"].copy()
        Y_O_new.insert(insert_at, query_result)
        u_dict["Y_O_real"] = Y_O_new
        return u_dict

    def query_dict_update(self, human_index):
        self.update_dict_with_vote(
            self.test_dict, 
            human_index
        )
        self.hyp_dict = self.test_dict.copy()
        return self.test_dict

    def get_hypothetical_query_dict(self, human_index, vote):
        return self.update_dict_with_vote(
            self.hyp_dict.copy(), 
            human_index, 
            vote
        )