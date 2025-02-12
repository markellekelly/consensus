import pickle
import sys
import numpy as np
import pandas as pd

from online_modeling import run_experiment_with_threshold


def get_fnames(
        name, 
        start_point, 
        use_ts, 
        use_corr, 
        dist_shift, 
        shuffle_num,
        add=""
    ):
    '''
    from the input parameters, provide the corresponding data file to read in,
    an informative file name to save results at, and a unique model id 
    (used to avoid conflicts in stan temporary directories)
    ''' 

    model_id = name + '_st' + str(start_point) + "_sh" + str(shuffle_num)

    ts_str = "" if use_ts else "_nots"
    corr_str = "" if use_corr else "_nocorr"
    ds_str = "_ds" if dist_shift else ""

    results_save_dir = "results/" + name + "/"
    results_save_f = "start" + str(start_point) + ts_str + corr_str + ds_str
    if shuffle_num != "":
        results_save_f += "_shuf"+ shuffle_num

    results_path = results_save_dir + results_save_f + add
    data_file = 'data/{}/data{}{}.pickle'.format( 
        name, ds_str, shuffle_num
    )

    return data_file, results_path, model_id



def main(dataset_name, start_point, suffix, shuffle_num):

    use_temp_scaling = 1 # 0 or 1, default 1
    use_correlations = 1 # 0 or 1, default 1
    distribution_shift = 0 # 0 or 1, default 0
    sliding_window = None # None or int < 250 (50 used for experiments), default None


    data_file, result_path_str, model_id = get_fnames(
        dataset_name, 
        start_point,
        use_temp_scaling,
        use_correlations,
        distribution_shift,
        shuffle_num,
        add=suffix
    )

    with open(data_file, 'rb') as handle:
        data_dict = pickle.load(handle)

    thresholds = list(np.arange(0, 0.01, 0.005)) + \
        list(np.arange(0.01, 0.05, 0.01)) + \
        list(np.arange(0.025, 0.05, 0.3)) + \
        list(np.arange(0.05, 0.3, 0.6))

    max_retries = 3

    for threshold in thresholds:
        attempts = 0

        while attempts < max_retries:
            try:
                run_experiment_with_threshold(
                    data_dict, 
                    threshold,
                    model_id,
                    results_path = result_path_str + str(threshold) + ".csv",
                    start_point = start_point, 
                    use_temp_scaling=use_temp_scaling, 
                    use_correlations=use_correlations,
                    distribution_shift=distribution_shift,
                    sliding_window=sliding_window
                )
                attempts = max_retries
            except Exception as e:
                # handle rare numerical errors in Stan by re-initializing
                attempts += 1
                print(f"Failed {attempts}/{max_retries} for t={threshold}: {e}")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 run.py <dataset_name> <start_point> <suffix> <shuffle_num>")
        sys.exit(1)

    dataset_name = sys.argv[1] # options:  "nih", "cifar", "imagenet", "chaoyang"
    start_point = int(sys.argv[2]) # 0, 250, 500, ...
    suffix = sys.argv[3] # additional thresholds: "-at","-at2" eta: "-eta01"
    shuffle_num = sys.argv[4] # "", "1",  "2", or "3"

    main(dataset_name, start_point, suffix, shuffle_num)