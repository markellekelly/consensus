import pandas as pd
import numpy as np
import pickle
import calibration as cal

datasets = [
    {
        "name" : "nih",
        "n_experts" : 5,
        "n_models" : 1,
        "k" : 2,
        "create_shuffles": True,
        "dist_shift": False
    },
    {
        "name" : "chaoyang",
        "n_experts" : 3,
        "n_models" : 1,
        "k" : 4,
        "create_shuffles": True,
        "dist_shift": False
    },
    {
        "name" : "cifar",
        "n_experts" : 3,
        "n_models" : 1,
        "k" : 3,
        "create_shuffles": False,
        "dist_shift": False
    },
    {
        "name" : "imagenet",
        "n_experts" : 3,
        "n_models" : 1,
        "k" : 3,
        "create_shuffles": True,
        "dist_shift": True
    },
]

def create_data_dict(dataset_name, df_to_save, n_experts, n_models, k, ext=""):
    Y_H = np.array(df_to_save[['expert'+str(i+1) for i in range(n_experts)]]) + 1
    Y_M = np.array(df_to_save[['model_p'+str(i) for i in range(k)]])
    Y_M = Y_M.reshape((len(df_to_save), 1, k))
    
    # previously used for chaoyang--do we need this? if so let's add it to the preprocessing
    #row_sums = Y_M.sum(axis=2)
    #Y_M_normalized = Y_M / row_sums[:, np.newaxis]

    data_dict = {
        'Y_M' : Y_M,
        'Y_H' : Y_H.tolist(),
        'n_models' : n_models,
        'n_humans' : n_experts,
        'K' : k
    }
    
    file_name = dataset_name+'/data{}{}.pickle'.format(str(shuffle_num), ext)
    with open(file_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_infexp_data_dict(df_to_save, n_experts, n_models, k, start_point):
    Y_H_inf = np.array(df_to_save[['expert'+str(i+1) for i in range(n_experts)]])
    Y_H_inf = Y_H_inf.transpose()
    d_new = np.array(df_to_save['consensus'])
    Y_M_inf = np.array(df_to_save[['model_p'+str(i) for i in range(k)]])
    model_confs = np.array([Y_M_inf])
    model_preds = np.array([[np.argmax(i) for i in j] for j in model_confs])

    df_to_save['model_correct'] = df_to_save['model_pred_int']==df_to_save['consensus']
    model_perf = np.array([[df_to_save['model_correct'].mean()]])
    class_wise_perf = np.array(
        df_to_save.groupby(
            'consensus'
        ).aggregate(
            {'model_correct':'mean'}
        )['model_correct']
    )

    n_tests = 250
    infexp_dict = {
        'model_confs' : model_confs[:,start_point:start_point+n_tests],
        'model_preds' : model_preds[:,start_point:start_point+n_tests],
        'targets' : d_new[start_point:start_point+n_tests],
        'true_targets' : d_new[start_point:start_point+n_tests],
        'expert_preds' : Y_H_inf[:,start_point:start_point+n_tests],
        'chosen_models' : np.array([0]),
        'model_perf' : model_perf,
        'model_perf_per_class' : class_wise_perf
    }
    
    return infexp_dict


def create_infexp_data_dicts(
        dataset_name, 
        df_to_save, 
        n_start_points, 
        n_experts, 
        n_models, 
        k
    ):
    for start_point in range(0, 250*n_start_points, 250):
        infexp_dict = create_infexp_data_dict(
            df_to_save, 
            n_experts, 
            n_models, 
            k, 
            start_point
        )

        file_name = dataset_name+'/infexp/infexp{}s{}.pickle'.format(
            start_point, str(shuffle_num)
        )
        with open(file_name, 'wb') as handle:
            pickle.dump(infexp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_accuracies(df, n_experts, k):
    n = len(df)
    df['model_pred_int'] = df.apply(
        lambda x: np.argmax([x['model_p'+str(i)] for i in range(k)]), axis=1
    )
    df['model_correct'] = df['model_pred_int']==df['consensus']
    test_accuracy = np.mean(df['model_correct'])
    class_wise_accs = list(df.groupby('consensus').aggregate(
            {'model_correct':'mean'}
    )['model_correct'])
    print("classifier (overall): {}".format(test_accuracy))
    print("\t " + str(class_wise_accs))
    probs = np.array(df[['model_p'+str(i) for i in range(k)]])
    ece = cal.get_ece(probs, df['consensus'])
    cal_error = cal.get_calibration_error(probs, df['consensus'])
    print("\tECE = {}; cal error = {}".format(ece, cal_error))
    
    for e in range(n_experts):
        e_corr_col = 'expert{}_correct'.format(e+1)
        df[e_corr_col] = df['expert'+str(e+1)]==df['consensus']
        expert_acc = sum(df['expert'+str(e+1)]==df['consensus'])/n
        class_wise_accs = list(df.groupby('consensus').aggregate(
                {e_corr_col:'mean'}
        )[e_corr_col])
        print ("expert {}: {}".format(e+1, expert_acc))
        print("\t " + str(class_wise_accs))
    print()
    print()


def main():
    for dat in datasets:
        dataset_name = dat['name']
        print('processing '+ dataset_name)
        df = pd.read_csv(dataset_name+'/data_clean.csv')
        
        n_experts, n_models, k = dat['n_experts'], dat['n_models'], dat['k']
        get_accuracies(df, n_experts, k)
        
        if dat["create_shuffles"]:

            for shuffle_num in ["", 1,2,3]:
                if shuffle_num == "":
                    df_to_save = df
                else:
                    df_to_save = df.sample(frac=1, random_state=shuffle_num)

                # create & save data dict for our method
                create_data_dict(dataset_name, df_to_save, n_experts, n_models, k)

                # create & save data dicts for infexp method
                n_start_points=3
                create_infexp_data_dicts(
                    dataset_name, 
                    df_to_save, 
                    n_start_points, 
                    n_experts, 
                    n_models, 
                    k
                )
                        
        else:
            create_data_dict(dataset_name, df, n_experts, n_models, k)
                
            n_start_points=12
            create_infexp_data_dicts(
                dataset_name, 
                df_to_save, 
                n_start_points, 
                n_experts, 
                n_models, 
                k
            )
            
        if dat["dist_shift"]:
            assert dat["create_shuffles"] # otherwise not implemented
            dat_before_ds = pd.read_csv(dataset_name+"/data_before_ds_clean.csv")
            get_accuracies(dat_before_ds, n_experts, k)
            dat_after_ds = pd.read_csv(dataset_name+"/data_after_ds_clean.csv")
            get_accuracies(dat_after_ds, n_experts, k)
            
            for shuffle_num in ["", 1,2,3]:
                if shuffle_num == "":
                    dat_before_to_save = dat_before_ds
                    dat_after_to_save = dat_after_ds
                else:
                    dat_before_to_save = dat_before_ds.sample(
                        frac=1, random_state=shuffle_num
                    )
                    dat_after_to_save = dat_after_ds.sample(
                        frac=1, random_state=shuffle_num
                    )

                df_to_save = pd.concat([
                    dat_before_to_save[:125], 
                    dat_after_to_save[:125], 
                    dat_before_to_save[125:250], 
                    dat_after_to_save[125:250],
                    dat_before_to_save[250:],
                    dat_after_to_save[250:]
                ])

                create_data_dict(
                    dataset_name, 
                    df_to_save, 
                    n_experts, 
                    n_models, 
                    k, 
                    ext="_ds".format(str(shuffle_num))
                )


if __name__ == "__main__":
    main()